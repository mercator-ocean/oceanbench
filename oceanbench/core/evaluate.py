# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import importlib.util
import json
import os
import sys
from os import environ
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Callable
from urllib.request import urlopen

import boto3
import xarray

from oceanbench import metrics
from oceanbench.core import metrics as core_metrics
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.regions import RegionLike, RegionSpec, region_to_dict, resolve_region
from oceanbench.core.runtime_configuration import (
    RuntimeConfiguration,
    runtime_configuration_from_environment,
    set_runtime_configuration,
)
from oceanbench.core.schema_validation import validate_against_schema
from oceanbench.core.scores import (
    CLASS4_METRIC_FUNCTION_NAME,
    GRIDDED_METRIC_FUNCTION_NAMES,
    SCORES_SCHEMA_VERSION,
    METRIC_FUNCTION_NAMES,
    scores_document,
    scores_file_name,
    write_scores_file,
)
from oceanbench.core.scores_page import scores_page_instructions, uploaded_scores_file_url
from oceanbench.core.version import __version__


def _parse_variable_environment(
    variable: str | None,
    environment_variable_name: OceanbenchEnvironmentVariable,
) -> str | None:
    return variable if variable else environ.get(environment_variable_name.value)


def _parse_input_non_manadatory(
    variable: str | None,
    environment_variable_name: OceanbenchEnvironmentVariable,
) -> str | None:
    return _parse_variable_environment(variable, environment_variable_name)


def _parse_input_mandatory(
    variable: str | None,
    environment_variable_name: OceanbenchEnvironmentVariable,
) -> str:
    parsed_variable = _parse_variable_environment(variable, environment_variable_name)
    if parsed_variable in (None, ""):
        raise Exception(
            f"Input {environment_variable_name.value} is mandatory for "
            + "OceanBench evaluation"
            + ", either as python parameter or environment variable"
        )
    return parsed_variable


def _resolve_region_input(region: RegionLike) -> RegionLike:
    if region is not None:
        return region
    return _parse_input_non_manadatory(
        None,
        OceanbenchEnvironmentVariable.OCEANBENCH_REGION,
    )


def resolve_evaluation_region(region: RegionLike) -> RegionSpec:
    return resolve_region(_resolve_region_input(region))


def read_challenger_python_code(
    challenger_python_code_uri_or_local_path: str,
) -> str:
    try:
        with open(challenger_python_code_uri_or_local_path, "r", encoding="utf8") as file:
            return file.read()
    except Exception:
        with urlopen(challenger_python_code_uri_or_local_path) as response:
            return response.read().decode("utf-8")


def _add_current_directory_to_python_path() -> None:
    # Same rule as IPython's InteractiveShellApp.init_path, so the challenger file imports what it imports in Jupyter.
    if "" in sys.path:
        return
    for index, path in enumerate(sys.path):
        if os.path.basename(path) in {"site-packages", "dist-packages"}:
            break
    else:
        index = 0
    sys.path.insert(index, "")


def open_challenger_dataset(challenger_python_code: str) -> xarray.Dataset:
    _add_current_directory_to_python_path()
    challenger_namespace: dict = {"__name__": "__main__"}
    exec(challenger_python_code, challenger_namespace)
    return challenger_namespace["challenger_dataset"]


def evaluate_challenger(
    challenger_python_code_uri_or_local_path: str | None = None,
    output_bucket: str | None = None,
    output_prefix: str | None = None,
    runtime_configuration: RuntimeConfiguration | None = None,
    region: RegionLike = None,
    viewer_artifacts_directory: str | None = None,
    viewer_only: bool = False,
    scores_file: str | None = None,
):
    """
    Compute all the benchmark scores for the given challenger dataset, by calling all functions of the `metrics` module.
    It runs the python code that opens the challenger dataset as `challenger_dataset: xarray.Dataset`, prints each score table and writes them to a scores file.

    This function is used for official evaluation.

    The scores file name is automatically derived from the challenger file name and the region:
    ``glonet.py`` becomes ``glonet.global.scores.json``.

    Parameters
    ----------
    challenger_python_code_uri_or_local_path : str, optional
        The python content that opens the challenger dataset. Required. Can be a remote file (URL), a DataURI or the path to a local file. Can also be configured with environment variable ``OCEANBENCH_CHALLENGER_PYTHON_CODE_URI_OR_LOCAL_PATH``.
    output_bucket : str, optional
        The destination S3 bucket of the scores file. If not provided, the scores file is written on the local filesystem. If provided, it is uploaded instead, using AWS S3 environment variables. Can also be configured with environment variable ``OCEANBENCH_OUTPUT_BUCKET``.
    output_prefix : str, optional
        The destination S3 prefix of the scores file. If ``output_bucket`` is not provided, this option is ignored. If provided, uses AWS S3 environment variables. Can also be configured with environment variable ``OCEANBENCH_OUTPUT_PREFIX``.
    runtime_configuration : RuntimeConfiguration, optional
        Runtime settings applied to the evaluation, including staging and remote retry behavior.
    region : str, RegionSpec, or None, optional
        The OceanBench region to evaluate on. This can be an official region id such as ``global`` or ``ibi``, or a custom ``oceanbench.regions.RegionSpec`` value.
    viewer_artifacts_directory : str, optional
        A local directory to also write the viewer data of the challenger to, after the scores file. The directory mirrors the viewer data prefix and can hold several challengers and regions. Needs the optional ``viewer`` extra (``pip install oceanbench[viewer]``). Can also be configured with environment variable ``OCEANBENCH_VIEWER_ARTIFACTS_DIR``.
    viewer_only : bool, optional
        Build only the map viewer data, for a challenger that was already scored: the score rows are read from the scores file of a previous evaluation, only the Class IV match-ups are recomputed, the other metrics are skipped and no scores file is written. Needs ``viewer_artifacts_directory``.
    scores_file : str, optional
        With ``viewer_only``, the local scores file to read. By default, the scores file of the challenger and region is read from ``output_bucket``/``output_prefix`` if a bucket is given, else from the current directory.
    """  # noqa
    resolved_challenger_python_code_uri_or_local_path = _parse_input_mandatory(
        challenger_python_code_uri_or_local_path,
        OceanbenchEnvironmentVariable.OCEANBENCH_CHALLENGER_PYTHON_CODE_URI_OR_LOCAL_PATH,
    )
    resolved_region = resolve_evaluation_region(region)
    resolved_output_bucket = _parse_input_non_manadatory(
        output_bucket,
        OceanbenchEnvironmentVariable.OCEANBENCH_OUTPUT_BUCKET,
    )
    resolved_output_prefix = _parse_input_non_manadatory(
        output_prefix,
        OceanbenchEnvironmentVariable.OCEANBENCH_OUTPUT_PREFIX,
    )
    resolved_viewer_artifacts_directory = _parse_input_non_manadatory(
        viewer_artifacts_directory,
        OceanbenchEnvironmentVariable.OCEANBENCH_VIEWER_ARTIFACTS_DIR,
    )
    resolved_runtime_configuration = runtime_configuration or runtime_configuration_from_environment()
    if viewer_only:
        if not resolved_viewer_artifacts_directory:
            raise ValueError("viewer_only needs a viewer artifacts directory")
        _build_viewer_data_only(
            resolved_challenger_python_code_uri_or_local_path,
            resolved_output_bucket,
            resolved_output_prefix,
            resolved_runtime_configuration,
            resolved_region,
            resolved_viewer_artifacts_directory,
            scores_file,
        )
        return
    elif scores_file:
        raise ValueError("scores_file is only used with viewer_only")
    _evaluate_challenger(
        resolved_challenger_python_code_uri_or_local_path,
        resolved_output_bucket,
        resolved_output_prefix,
        resolved_runtime_configuration,
        resolved_region,
        resolved_viewer_artifacts_directory,
    )


# Gridded metrics against the two references, computed in one pass so that the challenger is read once for both,
# and its mixed layer depth and geostrophic currents computed once for both.
# Not the variables: their two passes together keep the 1/12 degree challenger in memory (12 GB for one start).
GRIDDED_METRICS_COMPUTED_TOGETHER = [
    [
        f"rmsd_of_{quantity}_compared_to_glorys_reanalysis",
        f"rmsd_of_{quantity}_compared_to_glo12_analysis",
    ]
    for quantity in ("mixed_layer_depth", "geostrophic_currents")
]


def _metrics_computed_together(metric_function_name: str) -> list[str] | None:
    """The group of the metric, in the order of ``METRIC_FUNCTION_NAMES``, if it is the first of its group."""
    for group in GRIDDED_METRICS_COMPUTED_TOGETHER:
        if all(name in METRIC_FUNCTION_NAMES for name in group) and metric_function_name == min(
            group, key=METRIC_FUNCTION_NAMES.index
        ):
            return sorted(group, key=METRIC_FUNCTION_NAMES.index)
    return None


def _timed_step(timings: dict[str, float], step: str, function: Callable):
    step_start = perf_counter()
    result = function()
    timings[step] = perf_counter() - step_start
    return result


def format_evaluation_timings(timings: dict[str, float], scores_name: str) -> str:
    lines = [f"OceanBench step timings for {scores_name}:"]
    lines += [f"{duration:10.1f} s  {step}" for step, duration in timings.items()]
    return "\n".join(lines)


def _upload_scores_document(document: dict, output_file_name: str, output_bucket: str, output_name: str) -> None:
    with TemporaryDirectory() as temporary_directory:
        temporary_file_name = str(Path(temporary_directory) / output_file_name)
        write_scores_file(document, temporary_file_name)
        _upload_scores_file(temporary_file_name, output_bucket, output_name)


def _upload_scores_file(scores_file_name: str, output_bucket: str, output_name: str) -> None:
    endpoint_url = environ.get("BOTO3_ENDPOINT_URL", f"https://{environ['AWS_S3_ENDPOINT']}")
    boto3.client("s3", endpoint_url=endpoint_url).upload_file(
        scores_file_name,
        output_bucket,
        output_name,
        ExtraArgs={"ContentType": "application/json"},
    )


def _evaluate_challenger(
    challenger_python_code_uri_or_local_path: str,
    output_bucket: str | None,
    output_prefix: str | None,
    runtime_configuration: RuntimeConfiguration,
    region: RegionSpec,
    viewer_artifacts_directory: str | None = None,
):
    if viewer_artifacts_directory:
        require_viewer_extra()
    set_runtime_configuration(runtime_configuration)
    evaluation_start = perf_counter()
    timings: dict[str, float] = {}
    challenger_python_code = read_challenger_python_code(challenger_python_code_uri_or_local_path)
    challenger_dataset = _timed_step(
        timings,
        "open_challenger_dataset",
        lambda: open_challenger_dataset(challenger_python_code),
    )
    tables = {}
    start_tables = {}
    matchups = None
    together_results = {}
    for metric_function_name in METRIC_FUNCTION_NAMES:
        together = _metrics_computed_together(metric_function_name)
        if metric_function_name in together_results:
            # Computed with the first metric of its group, which carries the time of the group.
            timings[metric_function_name] = 0.0
            result = together_results.pop(metric_function_name)
            if viewer_artifacts_directory:
                tables[metric_function_name], start_tables[metric_function_name] = result
            else:
                tables[metric_function_name] = result
        elif together:
            together_results.update(
                _timed_step(
                    timings,
                    metric_function_name,
                    lambda: core_metrics._gridded_rmsd_metrics_together(
                        together, challenger_dataset, region=region, keep_start_tables=bool(viewer_artifacts_directory)
                    ),
                )
            )
            result = together_results.pop(metric_function_name)
            if viewer_artifacts_directory:
                tables[metric_function_name], start_tables[metric_function_name] = result
            else:
                tables[metric_function_name] = result
        elif viewer_artifacts_directory and metric_function_name == CLASS4_METRIC_FUNCTION_NAME:
            tables[metric_function_name], matchups = _timed_step(
                timings,
                metric_function_name,
                lambda: core_metrics.class4_rmsd_and_matchups_compared_to_observations(challenger_dataset, region),
            )
        elif viewer_artifacts_directory and metric_function_name in GRIDDED_METRIC_FUNCTION_NAMES:
            # The viewer bootstraps over the start dates, so it keeps the score of each one.
            tables[metric_function_name], start_tables[metric_function_name] = _timed_step(
                timings,
                metric_function_name,
                lambda: getattr(core_metrics, metric_function_name)(
                    challenger_dataset, region=region, keep_start_tables=True
                ),
            )
        else:
            tables[metric_function_name] = _timed_step(
                timings,
                metric_function_name,
                lambda: getattr(metrics, metric_function_name)(challenger_dataset, region=region),
            )
        print(f"\n{metric_function_name}\n{tables[metric_function_name].to_string()}")
    timings["total"] = perf_counter() - evaluation_start

    challenger_name = PurePosixPath(challenger_python_code_uri_or_local_path).stem
    output_file_name = scores_file_name(challenger_name, region)
    document = scores_document(challenger_name, challenger_python_code, challenger_dataset, region, tables, timings)
    output_name = f"{output_prefix}/{output_file_name}" if output_prefix else output_file_name
    if output_bucket:
        _upload_scores_document(document, output_file_name, output_bucket, output_name)
    else:
        write_scores_file(document, output_file_name)
    print(format_evaluation_timings(timings, output_name))
    uploaded_url = uploaded_scores_file_url(output_bucket, output_name) if output_bucket else None
    print(scores_page_instructions(output_file_name, uploaded_url))
    if viewer_artifacts_directory:
        _build_viewer_data(
            viewer_artifacts_directory, challenger_name, challenger_dataset, region, matchups, document, start_tables
        )


VIEWER_EXTRA_MODULES = ("pyarrow",)


def require_viewer_extra() -> None:
    """Fail early, before any scoring, when the optional ``viewer`` extra is not installed."""
    missing = [name for name in VIEWER_EXTRA_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(
            "Building the viewer data (--viewer-artifacts-dir or OCEANBENCH_VIEWER_ARTIFACTS_DIR) needs the optional "
            f"'viewer' extra, missing: {', '.join(missing)}. Install it with: pip install \"oceanbench[viewer]\""
        )


def _build_viewer_data(
    viewer_artifacts_directory: str,
    challenger_name: str,
    challenger_dataset: xarray.Dataset,
    region: RegionSpec,
    matchups,
    document: dict,
    start_tables: dict | None = None,
) -> None:
    # Imported here so that scoring alone never loads the viewer builder and its optional dependencies.
    from oceanbench.publish.viewer_data import build_viewer_data

    viewer_start = perf_counter()
    build_viewer_data(
        viewer_artifacts_directory,
        challenger_name,
        challenger_dataset,
        region,
        matchups,
        document["rows"],
        start_tables,
    )
    viewer_duration = perf_counter() - viewer_start
    print(f"Viewer data for {challenger_name} written to {viewer_artifacts_directory} in {viewer_duration:.1f} s")


def _read_scores_document(
    challenger_name: str,
    region: RegionSpec,
    output_bucket: str | None,
    output_prefix: str | None,
    scores_file: str | None,
) -> dict:
    output_file_name = scores_file_name(challenger_name, region)
    if scores_file:
        location = scores_file
    elif output_bucket:
        output_name = f"{output_prefix}/{output_file_name}" if output_prefix else output_file_name
        location = f"s3://{output_bucket}/{output_name}"
    else:
        location = output_file_name
    try:
        if scores_file or not output_bucket:
            with open(location, encoding="utf-8") as file:
                document = json.load(file)
        else:
            endpoint_url = environ.get("BOTO3_ENDPOINT_URL") or f"https://{environ['AWS_S3_ENDPOINT']}"
            response = boto3.client("s3", endpoint_url=endpoint_url).get_object(Bucket=output_bucket, Key=output_name)
            document = json.loads(response["Body"].read())
    except FileNotFoundError:
        raise ValueError(f"No scores file at {location}: run evaluate without viewer_only first") from None
    try:
        validate_against_schema(document, f"scores-v{SCORES_SCHEMA_VERSION}")
    except Exception as error:
        raise ValueError(f"{location} is not a valid scores file: {error}") from error
    expected = {
        "challenger": challenger_name,
        "region": region_to_dict(region),
        "oceanbench_version": __version__,
    }
    for field, value in expected.items():
        if document[field] != value:
            raise ValueError(
                f"{location} does not match this evaluation: {field} is {document[field]!r}, expected {value!r}"
            )
    return document


def _build_viewer_data_only(
    challenger_python_code_uri_or_local_path: str,
    output_bucket: str | None,
    output_prefix: str | None,
    runtime_configuration: RuntimeConfiguration,
    region: RegionSpec,
    viewer_artifacts_directory: str,
    scores_file: str | None,
) -> None:
    require_viewer_extra()
    set_runtime_configuration(runtime_configuration)
    challenger_name = PurePosixPath(challenger_python_code_uri_or_local_path).stem
    # Checked before opening the dataset, so a missing or stale scores file fails fast.
    document = _read_scores_document(challenger_name, region, output_bucket, output_prefix, scores_file)
    challenger_python_code = read_challenger_python_code(challenger_python_code_uri_or_local_path)
    challenger_dataset = open_challenger_dataset(challenger_python_code)
    _, matchups = core_metrics.class4_rmsd_and_matchups_compared_to_observations(challenger_dataset, region)
    _build_viewer_data(viewer_artifacts_directory, challenger_name, challenger_dataset, region, matchups, document)
