# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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


def _run_validate_nrt(args: argparse.Namespace) -> int:
    try:
        from oceanbench.core.nrt_validation import validate_nrt_forecast

        result, manifest_path_or_url = validate_nrt_forecast(
            system_id=args.system_id,
            system_label=args.system_label,
            forecast_zarr_template=args.forecast_zarr_template,
            observation_zarr_template=args.observation_zarr_template,
            forecast_init=args.forecast_init,
            observation_cutoff=args.observation_cutoff,
            octo_script=args.octo_script,
            octo_python=args.octo_python,
            octo_forecast_output_prefix=args.octo_forecast_output_prefix,
            skip_forecast_generation=args.skip_forecast_generation,
            forecast_temporary=args.forecast_temporary,
            forecast_ready_timeout_seconds=args.forecast_ready_timeout_seconds,
            forecast_ready_poll_seconds=args.forecast_ready_poll_seconds,
            cleanup_forecast_after_success=not args.keep_nrt_forecast,
            output_bucket=args.output_bucket,
            output_prefix=args.output_prefix,
            manifest_path=args.manifest_path,
            runtime_configuration=_runtime_configuration_from_args(args),
            region=_resolve_region_argument(args),
        )
    except Exception as error:
        print(f"FAIL: NRT forecast validation: {error}", file=sys.stderr)
        return 1

    print(f"{result.status}: {result.system_label} {result.forecast_init}")
    print(f"Observation cutoff: {result.observation_cutoff}")
    print(f"Forecast: {result.forecast_url}")
    print(f"Manifest: {manifest_path_or_url}")
    return 0 if result.status == "Complete" else 1


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
    validate_nrt_parser = subparsers.add_parser(
        "validate-nrt",
        help="Validate a near-real-time forecast against recent Class IV observations",
        description="Probe recent Class IV observations, request an Octo forecast, and run the NRT OceanBench report.",
    )
    validate_nrt_parser.add_argument(
        "--system-id",
        default="octo-glonet-p1d",
        help="Octo system identifier",
    )
    validate_nrt_parser.add_argument(
        "--system-label",
        default="GLONET",
        help="Human-readable system label used in reports",
    )
    validate_nrt_parser.add_argument(
        "--forecast-zarr-template",
        default=(
            "https://minio.dive.edito.eu/project-moiai-octo/public/octo/v0/ai-gallery/"
            "octo-glonet-p1d/{date}/{date}.zarr"
        ),
        help="Forecast Zarr URL template. Supports {compact_date}, {date}, {day}, {yyyymmdd}, and {YYYYMMDD}.",
    )
    validate_nrt_parser.add_argument(
        "--observation-zarr-template",
        default=None,
        help="Class IV observation Zarr URL template. Defaults to the OceanBench live observation template.",
    )
    validate_nrt_parser.add_argument(
        "--forecast-init",
        required=True,
        help=(
            "Forecast initialization day in YYYY-MM-DD format, resolved from the "
            "observation availability manifest."
        ),
    )
    validate_nrt_parser.add_argument(
        "--observation-cutoff",
        required=True,
        help="Complete observation day in YYYY-MM-DD format, resolved from the observation availability manifest.",
    )
    validate_nrt_parser.add_argument(
        "--octo-script",
        default=None,
        help="Path to Octo's orchestration_job.py. Required unless --skip-forecast-generation is used.",
    )
    validate_nrt_parser.add_argument(
        "--octo-python",
        default=None,
        help="Python executable used to run Octo. Defaults to the current Python executable.",
    )
    validate_nrt_parser.add_argument(
        "--octo-forecast-output-prefix",
        default=None,
        help=(
            "S3 key prefix, without bucket, where Octo should write the temporary "
            "NRT forecast. Defaults to Octo's dedicated OceanBench NRT prefix."
        ),
    )
    validate_nrt_parser.add_argument(
        "--skip-forecast-generation",
        action="store_true",
        help="Do not call Octo; only wait for and evaluate the forecast URL derived from --forecast-zarr-template.",
    )
    validate_nrt_parser.add_argument(
        "--forecast-temporary",
        action="store_true",
        help=(
            "Mark the forecast Zarr as temporary and delete it after a successful "
            "evaluation unless --keep-nrt-forecast is used."
        ),
    )
    validate_nrt_parser.add_argument(
        "--keep-nrt-forecast",
        action="store_true",
        help="Keep the temporary NRT forecast Zarr after successful evaluation.",
    )
    validate_nrt_parser.add_argument(
        "--forecast-ready-timeout-seconds",
        type=int,
        default=3600,
        help="Maximum time to wait for the forecast Zarr _SUCCESS marker.",
    )
    validate_nrt_parser.add_argument(
        "--forecast-ready-poll-seconds",
        type=int,
        default=60,
        help="Polling interval for the forecast Zarr _SUCCESS marker.",
    )
    validate_nrt_parser.add_argument(
        "--output-bucket",
        default=None,
        help="S3 bucket for output notebook and manifest.",
    )
    validate_nrt_parser.add_argument(
        "--output-prefix",
        default=None,
        help="S3 prefix for output notebook and manifest.",
    )
    validate_nrt_parser.add_argument(
        "--manifest-path",
        default=None,
        help="Local manifest path when --output-bucket is not used.",
    )
    validate_nrt_parser.add_argument(
        "--stage",
        action="append",
        choices=["challenger", "references", "observations", "all"],
        help="Stage selected datasets locally before evaluation. Repeat the flag to enable multiple staging targets.",
    )
    validate_nrt_parser.add_argument(
        "--stage-dir",
        default=None,
        help="Directory used for local staging when --stage is enabled",
    )
    validate_nrt_parser.add_argument(
        "--stage-max-workers",
        type=int,
        default=None,
        help="Maximum number of worker threads used to build local stage data",
    )
    validate_nrt_parser.add_argument(
        "--remote-retries",
        type=int,
        default=None,
        help="Number of retries for transient remote data read failures",
    )
    validate_nrt_parser.add_argument(
        "--keep-stage",
        action="store_true",
        help="Keep staged data after a successful validation command",
    )
    validate_region_group = validate_nrt_parser.add_mutually_exclusive_group()
    validate_region_group.add_argument(
        "--region",
        choices=get_pre_defined_region_names(),
        default=None,
        help="Official OceanBench region to evaluate on",
    )
    validate_region_group.add_argument(
        "--region-file",
        default=None,
        help="Path to a JSON file describing a custom evaluation region",
    )
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
    if args.command == "validate-nrt":
        sys.exit(_run_validate_nrt(args))


if __name__ == "__main__":
    main()
