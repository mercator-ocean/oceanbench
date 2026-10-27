# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from urllib.request import Request, urlopen

from oceanbench.core.local_stage import cleanup_local_stage_directory
from oceanbench.core.regions import RegionLike, get_pre_defined_region_names, load_region_file
from oceanbench.core.runtime_configuration import (
    STAGE_ALL_KEY,
    RuntimeConfiguration,
    runtime_configuration_from_environment,
    set_runtime_configuration,
)
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
    with ProcessPoolExecutor(
        max_workers=max_workers, max_tasks_per_child=1
    ) as executor:
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
            tuple(args.stage)
            if args.stage is not None
            else environment_configuration.staged_components
        ),
        stage_directory=args.stage_dir or environment_configuration.stage_directory,
        stage_max_workers=(
            args.stage_max_workers
            if args.stage_max_workers is not None
            else environment_configuration.stage_max_workers
        ),
        remote_retries=(
            args.remote_retries
            if args.remote_retries is not None
            else environment_configuration.remote_retries
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
    if (
        runtime_configuration.has_local_stage()
        and runtime_configuration.stage_directory is None
        and not args.keep_stage
        and all(result.success for result in results)
    ):
        cleanup_local_stage_directory(runtime_configuration.resolved_stage_directory())
    _print_results(results)
    return 0 if all(result.success for result in results) else 1


def _run_stage(args: argparse.Namespace) -> int:
    try:
        runtime_configuration = _runtime_configuration_from_args(args)
        region = _resolve_region_argument(args)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    from oceanbench.core.stage import stage_challenger_assets

    stage_challenger_assets(args.challenger, runtime_configuration, region)
    print(f"Staged data for {args.challenger} in {runtime_configuration.resolved_stage_directory()}")


def _validated_swot_error_dates(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    runs_start_date: str,
    runs_year: int,
) -> tuple[str | None, str | None]:
    if args.start_error_date is None:
        if args.end_error_date is not None:
            parser.error("--end-error-date requires --start-error-date")
        if args.plot_maps or args.maps_only:
            parser.error("--plot-maps / --maps-only require --start-error-date")
        return None, None
    try:
        start = datetime.strptime(args.start_error_date, "%Y-%m-%d")
        end = datetime.strptime(
            args.end_error_date or args.start_error_date, "%Y-%m-%d"
        )
    except ValueError:
        parser.error("--start-error-date / --end-error-date must be dates (YYYY-MM-DD)")
    if end < start:
        parser.error("--end-error-date must not be before --start-error-date")
    # Every day from the first run to the end of the runs year has a run
    # started at or before it ; the end of the period is checked precisely
    # against the lead days of the model by run_swot_validation.
    covered_start = datetime.strptime(runs_start_date, "%Y-%m-%d")
    covered_end = datetime(runs_year, 12, 31)
    if not covered_start <= start <= covered_end:
        parser.error(
            f"--start-error-date must be a day covered by a {runs_year} run "
            f"({runs_start_date} -> {covered_end:%Y-%m-%d})"
        )
    return args.start_error_date, args.end_error_date


def _run_swot_validate(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    from oceanbench.core.evaluate_model_against_swot import (
        DEFAULT_RUNS_YEAR,
        DEFAULT_START_DATE,
        MODELS,
        run_swot_validation,
    )

    unknown_models = [model for model in args.models if model not in MODELS]
    if unknown_models:
        print(
            f"Error: unknown model(s) {unknown_models}, available: {sorted(MODELS)}",
            file=sys.stderr,
        )
        return 1

    if args.subsample < 1:
        parser.error("--subsample must be >= 1")
    if args.vmax is not None and args.vmax <= 0:
        parser.error("--vmax must be > 0")

    start_error_date, end_error_date = _validated_swot_error_dates(
        args, parser, DEFAULT_START_DATE, DEFAULT_RUNS_YEAR
    )

    try:
        runtime_configuration = _runtime_configuration_from_args(args)
        region = _resolve_region_argument(args) or "global"
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    # The log file is created only once the arguments are validated.
    output_directory = args.output_dir
    os.makedirs(output_directory, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_directory, "validation.log")),
        ],
        force=True,
    )

    set_runtime_configuration(runtime_configuration)
    try:
        run_swot_validation(
            models=args.models,
            region=region,
            subsample=args.subsample,
            output_dir=output_directory,
            output=args.output,
            start_error_date=start_error_date,
            end_error_date=end_error_date,
            plot_maps=args.plot_maps,
            maps_only=args.maps_only,
            vmax=args.vmax,
        )
    except Exception as error:
        print(f"Error: swot-validate failed: {error}", file=sys.stderr)
        return 1

    # OceanBench never deletes a stage directory provided by the user
    # (--stage-dir) : only the default temporary one is cleaned up.
    if (
        runtime_configuration.has_local_stage()
        and runtime_configuration.stage_directory is None
        and not args.keep_stage
    ):
        cleanup_local_stage_directory(runtime_configuration.resolved_stage_directory())
    return 0


def _build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        prog="oceanbench",
        description="OceanBench CLI",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
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
        help=(
            "Stage selected datasets locally before evaluation. Repeat the flag to enable multiple staging targets."
            " Without it, --stage-dir stages all datasets."
        ),
    )
    _add_stage_arguments(evaluate_parser)
    evaluate_parser.add_argument(
        "--keep-stage",
        action="store_true",
        help="Keep the default temporary stage directory after a successful evaluate command",
    )
    _add_region_arguments(evaluate_parser)

    stage_parser = subparsers.add_parser(
        "stage",
        help="Download the data a challenger evaluation needs, without computing any score",
        description="Fill the stage directory so that a later evaluate run with the same --stage-dir needs no network.",
    )
    stage_parser.add_argument(
        "challenger",
        help="Challenger file path or URL to stage the data for",
    )
    _add_stage_arguments(stage_parser)
    _add_region_arguments(stage_parser)
    stage_parser.set_defaults(stage=[STAGE_ALL_KEY])

    # ==============================================================================

    swot_parser = subparsers.add_parser(
        "swot-validate",
        help="Validate models against SWOT L3 observations",
        description=(
            "Compute the RMSD Class IV table per lead day (CSV) between challenger models "
            "(glonet, glo12, xihe, wenhai, langya, hclimrep) and SWOT L3 observations, and "
            "optionally pointwise error outputs (error NetCDF + one error map per calendar day) "
            "for a requested period."
        ),
    )
    swot_parser.add_argument(
        "--models",
        nargs="+",
        default=["glo12"],
        help="Models to validate - available: glonet, glo12, xihe, wenhai, langya, hclimrep",
    )
    swot_parser.add_argument(
        "--subsample",
        type=int,
        default=10,
        help="Take 1 SWOT point over N",
    )
    swot_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help=(
            "Directory for all outputs (CSV, error NetCDF, maps, log), "
            "relative to the current directory (default: outputs)"
        ),
    )
    swot_parser.add_argument(
        "--output",
        type=str,
        default="rmsd_swot_results.csv",
        help="Name of the output CSV file (written inside --output-dir)",
    )
    swot_parser.add_argument(
        "--start-error-date",
        type=str,
        default=None,
        help=(
            "Calendar date (YYYY-MM-DD) from which pointwise error outputs are produced: "
            "error NetCDF + (with --plot-maps) one error map per calendar day. Without this "
            "option, only the RMSD table is computed. The RMSD table always covers all the "
            "weekly runs of 2024, independently of this option."
        ),
    )
    swot_parser.add_argument(
        "--end-error-date",
        type=str,
        default=None,
        help=(
            "Last calendar date of the pointwise error outputs, inclusive (YYYY-MM-DD). "
            "Defaults to --start-error-date."
        ),
    )
    swot_parser.add_argument(
        "--plot-maps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save one PNG error map per calendar day (requires cartopy : install oceanbench[maps])",
    )
    swot_parser.add_argument(
        "--maps-only",
        action="store_true",
        help="Skip the RMSD table and produce only the pointwise error outputs of the requested period",
    )
    swot_parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help=(
            "Fixed upper bound of the |error| color scale, so maps of different models are "
            "comparable (default: 99th percentile of the period)"
        ),
    )
    swot_parser.add_argument(
        "--stage",
        action="append",
        choices=["swot-observations", "all"],
        help=(
            "Stage the SWOT observations locally so that models with the same runs share one "
            "download. Repeat the flag to enable multiple staging targets."
        ),
    )
    _add_stage_arguments(swot_parser)
    swot_parser.add_argument(
        "--keep-stage",
        action="store_true",
        help="Keep staged data after a successful swot-validate command",
    )
    _add_region_arguments(swot_parser)

    return parser, evaluate_parser


def _add_stage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stage-dir",
        default=None,
        help="Directory used for local staging. Setting it turns staging on. OceanBench never deletes it",
    )
    parser.add_argument(
        "--stage-max-workers",
        type=int,
        default=None,
        help="Maximum number of worker threads used to build local stage data",
    )
    parser.add_argument(
        "--remote-retries",
        type=int,
        default=None,
        help="Number of attempts for each remote data read, retrying transient failures",
    )


def _add_region_arguments(parser: argparse.ArgumentParser) -> None:
    region_group = parser.add_mutually_exclusive_group()
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


def main():
    parser, evaluate_parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "evaluate":
        if args.all_challengers and args.challengers:
            evaluate_parser.error(
                "--all-challengers cannot be used with positional challenger arguments"
            )
        sys.exit(_run_evaluate(args))

    if args.command == "stage":
        sys.exit(_run_stage(args))

    if args.command == "swot-validate":
        sys.exit(_run_swot_validate(args, parser))


if __name__ == "__main__":
    main()
