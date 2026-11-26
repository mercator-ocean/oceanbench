# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Helpers of the EDITO process job: challenger file with a subset of start dates, S3 transfers, CPU limit,
scores summary for the log."""

import argparse
import base64
import gzip
import json
import math
import mimetypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

LAST_FILE_NAMES = {"datasets.json", "insights.json", "scores-summary.json", "index.json"}
ZARR_METADATA_FILE_NAMES = {".zmetadata", ".zarray", ".zattrs", ".zgroup", "zarr.json"}
UPLOAD_THREADS = 16


def _s3_client():
    import boto3

    endpoint = os.environ.get("BOTO3_ENDPOINT_URL") or os.environ["AWS_S3_ENDPOINT"]
    endpoint_url = endpoint if endpoint.startswith(("http://", "https://")) else f"https://{endpoint}"
    return boto3.client("s3", endpoint_url=endpoint_url)


def _read_text(path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://")):
        with urlopen(path_or_url) as response:
            return response.read().decode("utf-8")
    return Path(path_or_url).read_text(encoding="utf-8")


def _start_selection_code(start_dates: list[str], start_count: int | None) -> str:
    if start_dates:
        dates = ", ".join(f'"{date}"' for date in start_dates)
        return f"challenger_dataset = challenger_dataset.sel(first_day_datetime=pandas.to_datetime([{dates}]))\n"
    return f"challenger_dataset = challenger_dataset.isel(first_day_datetime=slice(0, {start_count}))\n"


def write_challenger(arguments: argparse.Namespace) -> None:
    """Copy the challenger code, followed by a selection of start dates when one is asked."""
    start_dates = [date.strip() for date in arguments.start_dates.split(",") if date.strip()]
    start_count = int(arguments.start_count) if arguments.start_count else None
    code = _read_text(arguments.source)
    if start_dates or start_count:
        code = (
            code.rstrip()
            + "\n\n# Start dates selected by the EDITO process job\nimport pandas\n\n"
            + _start_selection_code(start_dates, start_count)
            + "challenger_dataset\n"
        )
    Path(arguments.destination).write_text(code, encoding="utf-8")


def _existing_keys(client, bucket: str, prefix: str) -> list[str]:
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix.rstrip("/") + "/", MaxKeys=5)
    return [entry["Key"] for entry in response.get("Contents", [])]


def check_destination(arguments: argparse.Namespace) -> None:
    """Fail when the destination prefix already holds objects and overwriting is not allowed."""
    client = _s3_client()
    existing_keys = _existing_keys(client, arguments.bucket, arguments.prefix)
    if existing_keys and not arguments.overwrite:
        print(
            f"Destination s3://{arguments.bucket}/{arguments.prefix}/ is not empty (for example {existing_keys[0]})."
            " Choose another OUTPUT_PREFIX or set OVERWRITE=true.",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"Destination s3://{arguments.bucket}/{arguments.prefix}/ is writable for this job")


def _upload_rank(relative_path: PurePosixPath) -> int:
    """Data first, then zarr metadata, then the JSON files of subdirectories, then top level files, indexes last."""
    name = relative_path.name
    if name in LAST_FILE_NAMES or "manifest" in name or name.startswith("index"):
        return 4
    if len(relative_path.parts) == 1:
        return 3
    if name in ZARR_METADATA_FILE_NAMES:
        return 1
    if name.endswith(".json"):
        return 2
    return 0


def _content_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _upload_one(client, bucket: str, local_path: Path, key: str) -> None:
    client.upload_file(str(local_path), bucket, key, ExtraArgs={"ContentType": _content_type(local_path)})


def upload(arguments: argparse.Namespace) -> None:
    """Upload files and directories under the prefix, one rank after the other so that indexes come last."""
    client = _s3_client()
    prefix = arguments.prefix.strip("/")
    transfers = [
        (
            path,
            PurePosixPath(source.name) / path.relative_to(source).as_posix(),
            PurePosixPath(path.relative_to(source).as_posix()),
        )
        for source in map(Path, arguments.sources)
        if source.is_dir()
        for path in sorted(source.rglob("*"))
        if path.is_file()
    ] + [
        (source, PurePosixPath(source.name), PurePosixPath(source.name))
        for source in map(Path, arguments.sources)
        if source.is_file()
    ]
    ranks = sorted({_upload_rank(rank_path) for _, _, rank_path in transfers})
    for rank in ranks:
        batch = [(path, f"{prefix}/{key}") for path, key, rank_path in transfers if _upload_rank(rank_path) == rank]
        with ThreadPoolExecutor(max_workers=UPLOAD_THREADS) as executor:
            list(executor.map(lambda transfer: _upload_one(client, arguments.bucket, *transfer), batch))
        print(f"Uploaded {len(batch)} files (rank {rank}) to s3://{arguments.bucket}/{prefix}/")


def download(arguments: argparse.Namespace) -> None:
    """Fetch a file given as s3://bucket/key, an HTTPS URL or a local path."""
    destination = Path(arguments.destination)
    if arguments.source.startswith("s3://"):
        bucket, _, key = arguments.source[len("s3://") :].partition("/")
        _s3_client().download_file(bucket, key, str(destination))
    else:
        destination.write_text(_read_text(arguments.source), encoding="utf-8")


def _is_merge_input(relative_key: str) -> bool:
    """The files of a shard the merge reads: everything but its viewer data, except the eddy census."""
    return not relative_key.startswith("viewer/") or PurePosixPath(relative_key).name.startswith("eddies")


def download_shard(arguments: argparse.Namespace) -> None:
    """Fetch what the merge needs of a shard uploaded under ``prefix``."""
    client = _s3_client()
    prefix = arguments.prefix.strip("/") + "/"
    keys = [
        entry["Key"]
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=arguments.bucket, Prefix=prefix)
        for entry in page.get("Contents", [])
        if _is_merge_input(entry["Key"][len(prefix) :])
    ]
    if not keys:
        print(f"No shard at s3://{arguments.bucket}/{prefix}", file=sys.stderr)
        sys.exit(2)
    destination = Path(arguments.destination)

    def download_one(key: str) -> None:
        path = destination / key[len(prefix) :]
        path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(arguments.bucket, key, str(path))

    with ThreadPoolExecutor(max_workers=UPLOAD_THREADS) as executor:
        list(executor.map(download_one, keys))
    print(f"Downloaded {len(keys)} files of s3://{arguments.bucket}/{prefix} to {destination}")


def _cgroup_cpu_limit() -> float | None:
    cgroup_v2_file = Path("/sys/fs/cgroup/cpu.max")
    if cgroup_v2_file.exists():
        quota, period = cgroup_v2_file.read_text().split()[:2]
        return None if quota == "max" else int(quota) / int(period)
    quota_file, period_file = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_file.exists() and period_file.exists():
        quota = int(quota_file.read_text())
        return None if quota <= 0 else quota / int(period_file.read_text())
    return None


def cpu_limit(arguments: argparse.Namespace) -> None:
    """Print the usable CPU count: the cgroup quota, else the CPUs this process may run on."""
    limit = _cgroup_cpu_limit()
    affinity = len(os.sched_getaffinity(0))
    print(math.ceil(min(limit, affinity)) if limit else affinity)


def _metric_means(scores: dict) -> list[tuple[str, str, int, float]]:
    totals: dict[tuple[str, str], list[float]] = {}
    for row in scores.get("rows", []):
        value = row.get("value")
        if isinstance(value, (int, float)) and math.isfinite(value):
            totals.setdefault((row.get("metric", "?"), row.get("reference", "?")), []).append(float(value))
    return [(metric, reference, len(values), sum(values) / len(values)) for (metric, reference), values in totals.items()]


def print_scores(arguments: argparse.Namespace) -> None:
    """Print the scores file to the log: gzipped and base64-encoded when small enough, else a per-metric mean table."""
    raw = Path(arguments.scores_file).read_bytes()
    encoded = base64.b64encode(gzip.compress(raw)).decode()
    if len(encoded) <= arguments.max_bytes:
        print(f"scores file {Path(arguments.scores_file).name}: {len(raw)} bytes, {len(encoded)} bytes gzipped in base64")
        print("decode with: sed -n '/^BEGIN SCORES/,/^END SCORES/p' LOG | sed '1d;$d' | base64 -d | gunzip")
        print("BEGIN SCORES JSON GZIP BASE64")
        for start in range(0, len(encoded), 76):
            print(encoded[start : start + 76])
        print("END SCORES JSON GZIP BASE64")
        return
    print(f"scores file {Path(arguments.scores_file).name}: {len(encoded)} bytes encoded, too large for the log")
    print("BEGIN SCORES MEAN TABLE")
    print("metric\treference\tvalues\tmean")
    for metric, reference, count, mean in _metric_means(json.loads(raw)):
        print(f"{metric}\t{reference}\t{count}\t{mean:.6g}")
    print("END SCORES MEAN TABLE")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edito_job")
    commands = parser.add_subparsers(dest="command", required=True)
    challenger = commands.add_parser("write-challenger")
    challenger.add_argument("source")
    challenger.add_argument("destination")
    challenger.add_argument("--start-dates", default="")
    challenger.add_argument("--start-count", default="")
    challenger.set_defaults(function=write_challenger)
    destination = commands.add_parser("check-destination")
    destination.add_argument("bucket")
    destination.add_argument("prefix")
    destination.add_argument("--overwrite", action="store_true")
    destination.set_defaults(function=check_destination)
    upload_command = commands.add_parser("upload")
    upload_command.add_argument("bucket")
    upload_command.add_argument("prefix")
    upload_command.add_argument("sources", nargs="+")
    upload_command.set_defaults(function=upload)
    download_command = commands.add_parser("download")
    download_command.add_argument("source")
    download_command.add_argument("destination")
    download_command.set_defaults(function=download)
    shard = commands.add_parser("download-shard")
    shard.add_argument("bucket")
    shard.add_argument("prefix")
    shard.add_argument("destination")
    shard.set_defaults(function=download_shard)
    cpu = commands.add_parser("cpu-limit")
    cpu.set_defaults(function=cpu_limit)
    scores = commands.add_parser("print-scores")
    scores.add_argument("scores_file")
    scores.add_argument("--max-bytes", type=int, default=2_000_000)
    scores.set_defaults(function=print_scores)
    return parser


if __name__ == "__main__":
    parsed_arguments = _parser().parse_args()
    parsed_arguments.function(parsed_arguments)
