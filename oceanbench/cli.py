# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import argparse
import os
import sys
from pathlib import Path

from oceanbench.core.version import __version__

NEW_PIPELINE_NOTICE = "note: OceanBench 0.5 produces viewer artifacts and parquet scores instead of notebook reports."
DEFAULT_OUTPUT_DIRECTORY = "oceanbench-evaluation"
# Dask defaults to one thread per core. Several derived quantities hold float64 blocks per thread,
# so on a big shared node that default is a memory budget rather than a speed-up.
_DEFAULT_DASK_WORKER_CAP = 8
# Errors a user can fix from the message alone. Anything else keeps its traceback.
_USER_FACING_ERRORS = (ValueError, FileNotFoundError, NotADirectoryError, IsADirectoryError, PermissionError)


def _apply_runtime_configuration(args: argparse.Namespace) -> None:
    from oceanbench.core.runtime_configuration import (
        RuntimeConfiguration,
        runtime_configuration_from_environment,
        set_runtime_configuration,
    )

    environment_configuration = runtime_configuration_from_environment()
    set_runtime_configuration(
        RuntimeConfiguration(
            staged_components=(tuple(args.stage) if args.stage else environment_configuration.staged_components),
            stage_directory=(
                args.stage_dir if args.stage_dir is not None else environment_configuration.stage_directory
            ),
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
                args.cache_dir if args.cache_dir is not None else environment_configuration.local_cache_directory_path
            ),
        )
    )


def _apply_default_dask_concurrency() -> None:
    import dask

    from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable

    override = os.environ.get(OceanbenchEnvironmentVariable.OCEANBENCH_DASK_WORKERS.value)
    if not override and dask.config.get("num_workers", default=None) is not None:
        return
    worker_count = int(override) if override else min(_DEFAULT_DASK_WORKER_CAP, os.cpu_count() or 1)
    dask.config.set(num_workers=max(1, worker_count))


def _resolve_region_argument(args: argparse.Namespace) -> str | None:
    if args.region_file is None:
        return args.region
    if args.region is not None:
        raise ValueError("use either --region or --region-file, not both")
    from oceanbench.core.regions import load_region_file

    return load_region_file(args.region_file).id


def _target_name(target: str) -> str:
    name = Path(target.split("?", 1)[0].rstrip("/")).name
    return name.removesuffix(".py").removesuffix(".zarr") or target


def _resolve_targets(args: argparse.Namespace) -> list[str]:
    from oceanbench.runner.run import registered_challengers

    if args.all_challengers:
        return list(registered_challengers())
    return list(args.target)


def _run_evaluate(args: argparse.Namespace) -> int:
    from oceanbench.packs.evaluate import is_python_challenger_file

    if args.all_challengers and args.target:
        print("Error: --all-challengers cannot be combined with explicit targets", file=sys.stderr)
        return 1

    targets = _resolve_targets(args)
    if not targets:
        print("Error: provide an evaluation target or use --all-challengers", file=sys.stderr)
        return 1

    if args.all_challengers or any(is_python_challenger_file(target) for target in targets):
        print(NEW_PIPELINE_NOTICE, file=sys.stderr)

    try:
        _apply_runtime_configuration(args)
        region = _resolve_region_argument(args)
        offline_references = args.offline_references
    except _USER_FACING_ERRORS as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    output_directory = args.output if args.output is not None else DEFAULT_OUTPUT_DIRECTORY
    s3_bucket = args.s3_bucket if args.s3_bucket is not None else args.output_bucket
    s3_prefix = args.s3_prefix if args.s3_prefix is not None else args.output_prefix

    exit_code = 0
    for target in targets:
        if len(targets) > 1:
            print(f"\n== {target} ==")
        target_directory = output_directory if len(targets) == 1 else str(Path(output_directory) / _target_name(target))
        target_prefix = (
            s3_prefix if s3_prefix is None or len(targets) == 1 else f"{s3_prefix.rstrip('/')}/{_target_name(target)}"
        )
        exit_code |= _evaluate_one_target(
            args,
            target,
            region=region,
            offline_references=offline_references,
            output_directory=target_directory,
            s3_bucket=s3_bucket,
            s3_prefix=target_prefix,
        )

    _cleanup_stage(args, succeeded=exit_code == 0)
    return exit_code


def _cleanup_stage(args: argparse.Namespace, *, succeeded: bool) -> None:
    from oceanbench.core.local_stage import cleanup_local_stage_directory
    from oceanbench.core.runtime_configuration import current_runtime_configuration

    runtime_configuration = current_runtime_configuration()
    if runtime_configuration.has_local_stage() and not args.keep_stage and succeeded:
        cleanup_local_stage_directory(runtime_configuration.resolved_stage_directory())


def _evaluate_one_target(
    args: argparse.Namespace,
    target: str,
    *,
    region: str | None,
    offline_references: str | None,
    output_directory: str,
    s3_bucket: str | None,
    s3_prefix: str | None,
) -> int:
    from oceanbench.packs.evaluate import evaluate
    from oceanbench.packs.local_viewer import published_base_url

    try:
        result = evaluate(
            target,
            output_directory=output_directory,
            offline_references_directory=offline_references,
            region=region,
            year=args.year,
            published_scores_path=published_base_url() + "scores.parquet",
            published_challengers_path=published_base_url() + "challengers.json",
            metrics=args.metrics,
            viewer_artifacts=args.viewer_artifacts,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint=args.s3_endpoint,
            s3_env_file=args.s3_env_file,
        )
    except _USER_FACING_ERRORS as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    for flag in result.flags:
        print(f"note: {flag}", file=sys.stderr)
    if result.scores_path:
        print(f"scores:    {result.scores_path}")
        print(f"summary:   {result.summary_path}")
    if result.skill_baseline:
        print(f"skill vs:  {result.skill_baseline}")
    if result.scorecard_path:
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


def _add_evaluate_parser(subparsers: "argparse._SubParsersAction") -> None:
    from oceanbench.core.regions import GLOBAL_REGION_NAME, official_region_ids
    from oceanbench.packs.evaluate import DEFAULT_EVALUATION_YEAR, METRIC_NAMES
    from oceanbench.runner.run import registered_challengers

    parser = subparsers.add_parser(
        "evaluate",
        help="Score a forecast and write the scores the benchmark website reads",
        description=(
            "Score a forecast and emit the long-format records parquet plus the aggregated summary.\n\n"
            "TARGET is either your own forecast (a path or URL) or the slug of a challenger already in "
            "the benchmark. Your own forecast also gets a self-contained comparison scorecard laying it "
            "over the published challengers, so you can see where you stand before publishing.\n\n"
            "Forecast layout (weekly-store conventions, same as challengers): either a single combined "
            "zarr with dims (first_day_datetime, lead_day_index, depth, latitude, longitude) and the "
            "CF-named forecast variables, or a directory of weekly zarr stores named YYYYMMDD.zarr (one "
            "per forecast start, each with a 'time' lead-day dimension). See docs/local-evaluation.md.\n\n"
            "References and observations are read live from the public EDITO objects, so no download "
            "step is needed. Point --offline-references at a downloaded bundle to run without network.\n\n"
            "Known challenger slugs: " + ", ".join(registered_challengers())
        ),
        epilog=(
            "Published scores, challenger metadata, and viewer datasets default to the official "
            "OceanBench MinIO release. Set OCEANBENCH_PUBLISHED_BASE to override that base URL."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target",
        metavar="TARGET",
        nargs="*",
        help=(
            "Your forecast zarr (combined store or weekly-store directory), a known challenger slug, "
            "or a challenger .py file assigning challenger_dataset. Repeat to score several"
        ),
    )
    parser.add_argument(
        "--all-challengers",
        action="store_true",
        help="Score every registered challenger slug, each into its own output subdirectory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: ./oceanbench-evaluation)",
    )
    parser.add_argument(
        "--region",
        default=None,
        metavar="REGION",
        help=f"Region to score over (default: {GLOBAL_REGION_NAME}): " + ", ".join(official_region_ids()),
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help=f"Evaluation year (default: {DEFAULT_EVALUATION_YEAR})",
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
            "Also build the map viewer: its serving artifacts (Class-4 match-up parquet, eddy census, "
            "field pyramid, year-mode JSON) and a local viewer site you can open. Off by default"
        ),
    )
    parser.add_argument(
        "--offline-references",
        default=None,
        metavar="DIRECTORY",
        help=(
            "Read references and observations from a local reference directory instead of the "
            "live bucket. The directory's manifest fixes the region and year"
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
    parser.add_argument(
        "--region-file",
        default=None,
        metavar="PATH",
        help="JSON file describing the evaluation region, used instead of --region",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        metavar="PATH",
        help=(
            "Directory for the persistent chunk cache, so repeated runs re-read fetched chunks from "
            "disk (default: no cache; equivalent to OCEANBENCH_LOCAL_CACHE)"
        ),
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=["challenger", "references", "observations", "all"],
        help="Stage selected datasets locally before scoring. Repeat the flag for several targets",
    )
    parser.add_argument(
        "--stage-dir",
        default=None,
        metavar="DIRECTORY",
        help="Directory used for local staging when --stage is enabled",
    )
    parser.add_argument(
        "--stage-max-workers",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of worker threads used to build local stage data",
    )
    parser.add_argument(
        "--remote-retries",
        type=int,
        default=None,
        metavar="N",
        help="Number of retries for transient remote data read failures",
    )
    parser.add_argument(
        "--keep-stage",
        action="store_true",
        help="Keep staged data after a successful evaluate command",
    )
    # Accepted for 0.4.0 command lines. --output-bucket / --output-prefix are the old names of
    # --s3-bucket / --s3-prefix; --max-workers drove the notebook process pool and has no
    # equivalent on this route, which scores targets one after another.
    parser.add_argument("--output-bucket", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-prefix", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-workers", type=int, default=None, help=argparse.SUPPRESS)


def _run_view(args: argparse.Namespace) -> int:
    from oceanbench.publish.serve import build_viewer_server

    try:
        viewer = build_viewer_server(args.artifacts_directory, port=args.port)
    except _USER_FACING_ERRORS as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Error: cannot serve on port {args.port}: {error}", file=sys.stderr)
        return 1

    # serve_forever never returns, so the banner is flushed rather than left in the pipe buffer.
    print(f"viewer:    {viewer.viewer_directory}")
    print(f"artifacts: {viewer.artifacts_directory}")
    print(f"open:      {viewer.url}")
    print("Press Ctrl-C to stop.", flush=True)
    try:
        viewer.server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        viewer.server.server_close()
    return 0


def _add_view_parser(subparsers: "argparse._SubParsersAction") -> None:
    from oceanbench.publish.serve import DEFAULT_VIEWER_PORT

    parser = subparsers.add_parser(
        "view",
        help="Serve the map viewer over a local artifacts directory",
        description=(
            "Serve the OceanBench map viewer offline. The viewer single-page application "
            "(website/viewer, located relative to the installed package at the repository root) is "
            "served at '/', and ARTIFACTS_DIR is mounted at '/data/', so a locally produced viewer "
            "artifact tree is browsable without copying the application next to it.\n\n"
            "ARTIFACTS_DIR is the './data/' prefix of a viewer site: the directory holding "
            "datasets.json, insights.json and the per-dataset pyramid stores. "
            "'oceanbench evaluate --viewer-artifacts' writes one under its viewer/data directory.\n\n"
            "The server binds 127.0.0.1 only and no browser is opened; the URL to open is printed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("artifacts_directory", metavar="ARTIFACTS_DIR", help="Viewer artifacts directory to mount")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_VIEWER_PORT,
        help=f"Port to listen on (default: {DEFAULT_VIEWER_PORT}; 0 picks a free port)",
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


def _run_reconcile_viewer_artifacts(args: argparse.Namespace) -> int:
    from oceanbench.publish.reconcile import ReconciliationError, reconcile_viewer_artifacts

    try:
        reconcile_viewer_artifacts(
            args.artifacts_base,
            dataset=args.dataset,
            region=args.region,
            output_path=args.output,
            starts_per_variable=args.starts_per_variable,
            cells_per_variable=args.cells_per_variable,
        )
    except ReconciliationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001 - surface a clean message to the CLI user
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


def _add_reconcile_viewer_artifacts_parser(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(
        "reconcile-viewer-artifacts",
        help="Recompute headline numbers from published viewer artifacts and verify they match",
        description=(
            "Given a published viewer-artifact base (a local directory or an https:// prefix holding "
            "insights.json and scores-summary.json), recompute the Class-4 pooled RMSD, year per-start "
            "RMSD/bias and error-geography straight from the match-up parquet and assert they match the "
            "official aggregates, and structurally validate the eddy census. Writes a verification report "
            "JSON and exits non-zero on any mismatch."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("artifacts_base", help="Published viewer/data base (local directory or https:// prefix)")
    parser.add_argument("--dataset", default=None, help="Reconcile only this dataset slug (default: all)")
    parser.add_argument("--region", default=None, help="Reconcile only this region (default: all)")
    parser.add_argument("--output", default=None, help="Path for the verification report JSON")
    parser.add_argument("--starts-per-variable", type=int, default=4, help="Sampled year starts per variable")
    parser.add_argument("--cells-per-variable", type=int, default=20, help="Sampled geography cells per variable")


def _run_merge_realism(args: argparse.Namespace) -> int:
    from oceanbench.publish.merge_realism import merge_realism_scores

    try:
        result = merge_realism_scores(
            args.unit_directory,
            args.realism_directory,
            skill_baseline=args.skill_baseline,
        )
    except Exception as error:  # noqa: BLE001 - surface a clean message to the CLI user
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"{result.scores_path}: {result.realism_row_count} realism rows, {result.total_row_count} rows total")
    return 0


def _add_merge_realism_parser(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(
        "merge-realism",
        help="Merge a realism-only evaluation output into an already-scored unit output",
        description=(
            "Append the realism rows of a realism-only rerun (evaluate --metrics realism) to the "
            "scores.parquet of an already-scored unit, dropping any realism row already there so the "
            "merge is idempotent, and regenerate scores-summary.json and scores-<slug>.json through the "
            "same code path evaluate uses."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("unit_directory", help="Evaluation output directory to merge into")
    parser.add_argument("realism_directory", help="Output directory of the realism-only rerun")
    parser.add_argument(
        "--skill-baseline",
        default=None,
        help="Baseline slug skill is quoted against (default: read back from the unit's summary)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oceanbench",
        description="OceanBench CLI",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    _add_evaluate_parser(subparsers)
    _add_view_parser(subparsers)
    _add_publish_s3_parser(subparsers)
    _add_reconcile_viewer_artifacts_parser(subparsers)
    _add_merge_realism_parser(subparsers)
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    _apply_default_dask_concurrency()

    if args.command == "evaluate":
        sys.exit(_run_evaluate(args))

    if args.command == "view":
        sys.exit(_run_view(args))

    if args.command == "publish-s3":
        sys.exit(_run_publish_s3(args))

    if args.command == "reconcile-viewer-artifacts":
        sys.exit(_run_reconcile_viewer_artifacts(args))

    if args.command == "merge-realism":
        sys.exit(_run_merge_realism(args))


if __name__ == "__main__":
    main()
