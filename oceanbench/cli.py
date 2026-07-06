# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from oceanbench.core.local_stage import cleanup_local_stage_directory
from oceanbench.core.regions import RegionLike, get_pre_defined_region_names, load_region_file
from oceanbench.core.runtime_configuration import RuntimeConfiguration, runtime_configuration_from_environment
from oceanbench.core.version import __version__

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/mercator-ocean/oceanbench"
GITHUB_API_BASE = "https://api.github.com/repos/mercator-ocean/oceanbench/contents"
CHALLENGER_DIRECTORY = "challenger_datasets"


@dataclass(frozen=True)
class EvaluationResult:
    challenger: str
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


def _get_version_ref() -> str:
    return f"v{__version__}"


def _resolve_all_challenger_urls() -> list[str]:
    ref = _get_version_ref()
    url = f"{GITHUB_API_BASE}/{CHALLENGER_DIRECTORY}?ref={ref}"
    request = Request(url, headers={"Accept": "application/vnd.github.v3+json"})
    with urlopen(request) as response:
        entries = json.loads(response.read())
    return [
        f"{GITHUB_RAW_BASE}/{ref}/{CHALLENGER_DIRECTORY}/{entry['name']}"
        for entry in entries
        if entry["name"].endswith(".py")
    ]


def _resolve_region_argument(args: argparse.Namespace) -> RegionLike:
    if args.region_file is not None:
        return load_region_file(args.region_file)
    return args.region


def _evaluate_one(
    challenger: str,
    output_bucket: str | None,
    output_prefix: str | None,
    runtime_configuration: RuntimeConfiguration,
    region: RegionLike,
) -> EvaluationResult:
    try:
        from oceanbench.core.evaluate import evaluate_challenger

        evaluate_challenger(
            challenger_python_code_uri_or_local_path=challenger,
            output_bucket=output_bucket,
            output_prefix=output_prefix,
            runtime_configuration=runtime_configuration,
            region=region,
        )
        return EvaluationResult(challenger=challenger)
    except Exception as exception:
        return EvaluationResult(challenger=challenger, error=str(exception))


def _resolve_challengers(args: argparse.Namespace) -> list[str]:
    if args.all_challengers:
        return _resolve_all_challenger_urls()
    return args.challengers


def _evaluate_all(
    challengers: list[str],
    output_bucket: str | None,
    output_prefix: str | None,
    max_workers: int | None,
    runtime_configuration: RuntimeConfiguration,
    region: RegionLike,
) -> list[EvaluationResult]:
    # Notebook evaluations are heavy and can leave substantial state behind in a
    # worker process. Recycle the worker after each challenger to avoid
    # cross-challenger memory growth during `oceanbench evaluate a.py b.py ...`.
    with ProcessPoolExecutor(max_workers=max_workers, max_tasks_per_child=1) as executor:
        futures = {
            executor.submit(
                _evaluate_one,
                challenger,
                output_bucket,
                output_prefix,
                runtime_configuration,
                region,
            ): challenger
            for challenger in challengers
        }
        return [future.result() for future in as_completed(futures)]


def _print_results(results: list[EvaluationResult]) -> None:
    for result in results:
        if result.success:
            print(f"OK: {result.challenger}")
        else:
            print(f"FAIL: {result.challenger}: {result.error}", file=sys.stderr)

    successes = sum(1 for result in results if result.success)
    failures = sum(1 for result in results if not result.success)
    print(f"\n{successes} succeeded, {failures} failed")


def _runtime_configuration_from_args(args: argparse.Namespace) -> RuntimeConfiguration:
    environment_configuration = runtime_configuration_from_environment()
    return RuntimeConfiguration(
        staged_components=(
            tuple(args.stage) if args.stage is not None else environment_configuration.staged_components
        ),
        stage_directory=args.stage_dir if args.stage_dir is not None else environment_configuration.stage_directory,
        stage_max_workers=(
            args.stage_max_workers
            if args.stage_max_workers is not None
            else environment_configuration.stage_max_workers
        ),
        remote_retries=(
            args.remote_retries if args.remote_retries is not None else environment_configuration.remote_retries
        ),
        class4_fast_interpolation=environment_configuration.class4_fast_interpolation,
        local_cache_directory_path=(
            args.local_cache if args.local_cache is not None else environment_configuration.local_cache_directory_path
        ),
    )


def _run_evaluate(args: argparse.Namespace) -> int:
    challengers = _resolve_challengers(args)

    if not challengers:
        print(
            "Error: provide challenger files or use --all-challengers",
            file=sys.stderr,
        )
        return 1

    if args.max_workers is not None and args.max_workers < 1:
        print("Error: --max-workers must be >= 1", file=sys.stderr)
        return 1

    try:
        runtime_configuration = _runtime_configuration_from_args(args)
        region = _resolve_region_argument(args)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    results = _evaluate_all(
        challengers,
        args.output_bucket,
        args.output_prefix,
        args.max_workers,
        runtime_configuration,
        region,
    )
    if runtime_configuration.has_local_stage() and not args.keep_stage and all(result.success for result in results):
        cleanup_local_stage_directory(runtime_configuration.resolved_stage_directory())
    _print_results(results)
    return 0 if all(result.success for result in results) else 1


def _run_evaluate_local(args: argparse.Namespace) -> int:
    from oceanbench.packs.evaluate import evaluate_local
    from oceanbench.packs.local_viewer import published_base_url

    output_directory = args.output if args.output is not None else "oceanbench-local-evaluation"
    try:
        result = evaluate_local(
            args.forecasts,
            pack_directory=args.pack,
            output_directory=output_directory,
            published_scores_path=published_base_url() + "scores.parquet",
            published_challengers_path=published_base_url() + "challengers.json",
            metrics=args.metrics,
            artifacts=args.artifacts,
        )
    except Exception as error:  # noqa: BLE001 - surface a clean message to the CLI user
        print(f"Error: {error}", file=sys.stderr)
        return 1
    for flag in result.flags:
        print(f"note: {flag}", file=sys.stderr)
    if result.scores_path:
        print(f"scores:    {result.scores_path}")
        print(f"summary:   {result.summary_path}")
        print(f"scorecard: {result.scorecard_path}")
        print(f"\nOpen the scorecard locally (no server needed): file://{Path(result.scorecard_path).resolve()}")
    if result.viewer_directory:
        print(f"viewer:    {result.viewer_directory}")
        print(
            "serve:     " f"{sys.executable} -m http.server --directory {Path(result.viewer_directory).resolve()} 8799"
        )
        print("open:      http://127.0.0.1:8799/?data_base=local")
    return 0


def _add_evaluate_local_parser(subparsers: "argparse._SubParsersAction") -> None:
    from oceanbench.packs.evaluate import METRIC_NAMES

    parser = subparsers.add_parser(
        "evaluate-local",
        help="Score a local forecast against an evaluation pack and build an overlay scorecard",
        description=(
            "Score your own forecast zarr(s) against a downloaded evaluation pack (contracts.md §7) "
            "and emit the standard long-format records parquet, the aggregated summary, and a "
            "self-contained overlay scorecard laying your model over the published challengers.\n\n"
            "Forecast layout (weekly-store conventions, same as challengers): either a single combined "
            "zarr with dims (first_day_datetime, lead_day_index, depth, latitude, longitude) and the "
            "CF-named forecast variables, or a directory of weekly zarr stores named YYYYMMDD.zarr (one "
            "per forecast start, each with a 'time' lead-day dimension). See docs/local-evaluation.md."
        ),
        epilog=(
            "Published scores, challenger metadata, and viewer datasets default to the official "
            "OceanBench MinIO release. Set OCEANBENCH_PUBLISHED_BASE to override that base URL."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("forecasts", help="Path or URL to the forecast zarr (combined store or weekly-store directory)")
    parser.add_argument(
        "--pack", required=True, help="Path to the evaluation pack directory (contains pack-manifest.json)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: ./oceanbench-local-evaluation)",
    )
    parser.add_argument(
        "--artifacts",
        choices=("scores", "all"),
        default="scores",
        help="Artifacts to build: scores (default), or all (scores and viewer)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRIC_NAMES,
        default=None,
        metavar="M",
        help="Metrics to run (default: all): " + ", ".join(METRIC_NAMES),
    )


def _run_publish_s3(args: argparse.Namespace) -> int:
    from oceanbench.publish.s3 import build_upload_plan, content_type_for_path, upload_tree

    if args.dry_run:
        try:
            plan = build_upload_plan(args.local_root, args.prefix)
        except (NotADirectoryError, FileNotFoundError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        total_bytes = sum(item.size for item in plan)
        for item in plan:
            print(f"  {item.size:>12,}  {content_type_for_path(item.local_path):<24}  {item.key}")
        print(
            f"\ndry-run: {len(plan)} objects, {total_bytes:,} bytes "
            f"({total_bytes / 1e6:.1f} MB) -> s3://{args.bucket}/{args.prefix.strip('/')}/"
        )
        return 0

    try:
        summary = upload_tree(
            args.local_root,
            bucket=args.bucket,
            prefix=args.prefix,
            endpoint=args.endpoint,
            force=args.force,
            max_workers=args.max_workers,
            env_file=args.env_file,
        )
    except Exception as error:  # noqa: BLE001 - surface a clean message to the CLI user
        print(f"Error: {error}", file=sys.stderr)
        return 1

    rate = summary.uploaded_bytes / 1e6 / summary.elapsed_seconds if summary.elapsed_seconds > 0 else 0.0
    print(
        f"published to s3://{args.bucket}/{args.prefix.strip('/')}/ on {args.endpoint}\n"
        f"  uploaded: {summary.uploaded_count} objects, {summary.uploaded_bytes:,} bytes "
        f"({summary.uploaded_bytes / 1e6:.1f} MB)\n"
        f"  skipped:  {summary.skipped_count} objects (remote size already matched)\n"
        f"  planned:  {summary.planned_count} objects, {summary.total_bytes:,} bytes total\n"
        f"  elapsed:  {summary.elapsed_seconds:.1f}s ({rate:.1f} MB/s uploaded)"
    )
    return 0


def _add_publish_s3_parser(subparsers: "argparse._SubParsersAction") -> None:
    from oceanbench.publish.s3 import DEFAULT_MAX_WORKERS, EDITO_MINIO_ENDPOINT

    parser = subparsers.add_parser(
        "publish-s3",
        help="Upload a local benchmark catalog tree to S3-compatible object storage",
        description=(
            "Upload the local publish output (the catalog tree from the benchmark publish step) to "
            "s3://<bucket>/<prefix>/, preserving layout (contracts.md §8). Uploads run in parallel; objects "
            "whose remote size already matches the local size are skipped (idempotent) unless --force is given. "
            "Credentials resolve from AWS_* env vars if set, otherwise from an EDITO offline token "
            "(EDITO_MINIO_OFFLINE_TOKEN) minted into temporary STS credentials."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("local_root", help="Local directory containing the catalog tree to upload")
    parser.add_argument("--bucket", required=True, help="Target S3 bucket (e.g. project-oceanbench)")
    parser.add_argument(
        "--prefix",
        required=True,
        help="Target key prefix within the bucket (e.g. dev/benchmark/<release>)",
    )
    parser.add_argument(
        "--endpoint",
        default=EDITO_MINIO_ENDPOINT,
        help=f"S3-compatible endpoint URL (default: {EDITO_MINIO_ENDPOINT})",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Optional .env file to source the EDITO offline token from (AWS_* env vars still win)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Number of parallel upload threads (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload every object even when a same-size remote object already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the objects that would be uploaded without any network writes",
    )


def _build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        prog="oceanbench",
        description="OceanBench CLI",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate one or more challengers",
        description="Evaluate challengers against OceanBench metrics.",
    )
    evaluate_parser.add_argument(
        "challengers",
        nargs="*",
        help="Challenger file paths or URLs to evaluate",
    )
    evaluate_parser.add_argument(
        "--all-challengers",
        action="store_true",
        help="Evaluate all challengers from GitHub for the current version",
    )
    evaluate_parser.add_argument(
        "--output-bucket",
        default=None,
        help="S3 bucket for output notebooks",
    )
    evaluate_parser.add_argument(
        "--output-prefix",
        default=None,
        help="S3 prefix for output notebooks",
    )
    evaluate_parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker processes to use for evaluation",
    )
    evaluate_parser.add_argument(
        "--stage",
        action="append",
        choices=["challenger", "references", "observations", "all"],
        help="Stage selected datasets locally before evaluation. Repeat the flag to enable multiple staging targets.",
    )
    evaluate_parser.add_argument(
        "--stage-dir",
        default=None,
        help="Directory used for local staging when --stage is enabled",
    )
    evaluate_parser.add_argument(
        "--stage-max-workers",
        type=int,
        default=None,
        help="Maximum number of worker threads used to build local stage data",
    )
    evaluate_parser.add_argument(
        "--local-cache",
        default=None,
        help="Directory used to cache downloaded and computed datasets locally between runs",
    )
    evaluate_parser.add_argument(
        "--remote-retries",
        type=int,
        default=None,
        help="Number of retries for transient remote data read failures",
    )
    evaluate_parser.add_argument(
        "--keep-stage",
        action="store_true",
        help="Keep staged data after a successful evaluate command",
    )
    region_group = evaluate_parser.add_mutually_exclusive_group()
    region_group.add_argument(
        "--region",
        choices=get_pre_defined_region_names(),
        default=None,
        help="Official OceanBench region to evaluate on",
    )
    region_group.add_argument(
        "--region-file",
        default=None,
        help="Path to a JSON file describing a custom evaluation region",
    )

    _add_evaluate_local_parser(subparsers)
    _add_publish_s3_parser(subparsers)
    return parser, evaluate_parser


def main():
    parser, evaluate_parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "evaluate":
        if args.all_challengers and args.challengers:
            evaluate_parser.error("--all-challengers cannot be used with positional challenger arguments")
        sys.exit(_run_evaluate(args))

    if args.command == "evaluate-local":
        sys.exit(_run_evaluate_local(args))

    if args.command == "publish-s3":
        sys.exit(_run_publish_s3(args))


if __name__ == "__main__":
    main()
