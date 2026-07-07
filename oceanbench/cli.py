# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import argparse
import sys
from pathlib import Path

from oceanbench.core.version import __version__


def _run_evaluate(args: argparse.Namespace) -> int:
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
            viewer_artifacts=args.viewer_artifacts,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            s3_endpoint=args.s3_endpoint,
            s3_env_file=args.s3_env_file,
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
    if result.matchup_parquet_path:
        print(f"matchups:  {result.matchup_parquet_path}")
    if result.eddy_census_path:
        print(f"eddies:    {result.eddy_census_path}")
    if result.year_error_geography_path:
        print(f"year-geo:  {result.year_error_geography_path}")
        print(f"year-rmsd: {result.year_rmsd_by_start_path}")
    if result.published_prefix:
        print(f"published: {result.published_prefix}")
    return 0


def _add_evaluate_parser(
    subparsers: "argparse._SubParsersAction", name: str = "evaluate", *, hidden: bool = False
) -> None:
    from oceanbench.packs.evaluate import METRIC_NAMES

    parser = subparsers.add_parser(
        name,
        help=(
            argparse.SUPPRESS
            if hidden
            else "Score a local forecast against an evaluation pack and build an overlay scorecard"
        ),
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
    parser.add_argument(
        "--viewer-artifacts",
        action="store_true",
        help=(
            "Also produce the viewer serving artifacts (Class-4 match-up parquet, eddy census, "
            "field pyramid and year-mode JSON) under the output directory"
        ),
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help="Optional S3 bucket to upload the produced output tree to (e.g. project-oceanbench)",
    )
    parser.add_argument(
        "--s3-prefix",
        default=None,
        help="Target key prefix within --s3-bucket (required when --s3-bucket is given)",
    )
    parser.add_argument(
        "--s3-endpoint",
        default=None,
        help="S3-compatible endpoint URL (defaults to the EDITO MinIO endpoint)",
    )
    parser.add_argument(
        "--s3-env-file",
        default=None,
        help="Optional .env file to source the EDITO offline token from (AWS_* env vars still win)",
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oceanbench",
        description="OceanBench CLI",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="{evaluate,publish-s3}")

    _add_evaluate_parser(subparsers)
    _add_evaluate_parser(subparsers, "evaluate-local", hidden=True)
    subparsers._choices_actions = [choice for choice in subparsers._choices_actions if choice.dest != "evaluate-local"]
    _add_publish_s3_parser(subparsers)
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "evaluate":
        sys.exit(_run_evaluate(args))

    if args.command == "evaluate-local":
        print("note: 'oceanbench evaluate-local' is deprecated; use 'oceanbench evaluate'", file=sys.stderr)
        sys.exit(_run_evaluate(args))

    if args.command == "publish-s3":
        sys.exit(_run_publish_s3(args))


if __name__ == "__main__":
    main()
