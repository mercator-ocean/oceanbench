# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
import sys
from os import environ
from pathlib import PurePosixPath
from time import perf_counter
from typing import Callable
from urllib.request import urlopen

import boto3
import xarray

from oceanbench import metrics
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.regions import RegionLike, RegionSpec, resolve_region
from oceanbench.core.runtime_configuration import (
    RuntimeConfiguration,
    runtime_configuration_from_environment,
    set_runtime_configuration,
)
from oceanbench.core.scores import METRIC_FUNCTION_NAMES, scores_document, scores_file_name, write_scores_file


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
        The destination S3 bucket of the scores file. The scores file is always written on the local filesystem, and also uploaded when this is provided, using AWS S3 environment variables. Can also be configured with environment variable ``OCEANBENCH_OUTPUT_BUCKET``.
    output_prefix : str, optional
        The destination S3 prefix of the scores file. If ``output_bucket`` is not provided, this option is ignored. If provided, uses AWS S3 environment variables. Can also be configured with environment variable ``OCEANBENCH_OUTPUT_PREFIX``.
    runtime_configuration : RuntimeConfiguration, optional
        Runtime settings applied to the evaluation, including staging and remote retry behavior.
    region : str, RegionSpec, or None, optional
        The OceanBench region to evaluate on. This can be an official region id such as ``global`` or ``ibi``, or a custom ``oceanbench.regions.RegionSpec`` value.
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
    resolved_runtime_configuration = runtime_configuration or runtime_configuration_from_environment()
    _evaluate_challenger(
        resolved_challenger_python_code_uri_or_local_path,
        resolved_output_bucket,
        resolved_output_prefix,
        resolved_runtime_configuration,
        resolved_region,
    )


def _timed_step(timings: dict[str, float], step: str, function: Callable):
    step_start = perf_counter()
    result = function()
    timings[step] = perf_counter() - step_start
    return result


def format_evaluation_timings(timings: dict[str, float], scores_name: str) -> str:
    lines = [f"OceanBench step timings for {scores_name}:"]
    lines += [f"{duration:10.1f} s  {step}" for step, duration in timings.items()]
    return "\n".join(lines)


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
):
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
    for metric_function_name in METRIC_FUNCTION_NAMES:
        tables[metric_function_name] = _timed_step(
            timings,
            metric_function_name,
            lambda: getattr(metrics, metric_function_name)(challenger_dataset, region=region),
        )
        print(f"\n{metric_function_name}\n{tables[metric_function_name].to_string()}")
    timings["total"] = perf_counter() - evaluation_start

    challenger_name = PurePosixPath(challenger_python_code_uri_or_local_path).stem
    output_file_name = scores_file_name(challenger_name, region)
    write_scores_file(
        scores_document(challenger_name, challenger_python_code, challenger_dataset, region, tables, timings),
        output_file_name,
    )
    output_name = f"{output_prefix}/{output_file_name}" if output_prefix else output_file_name
    if output_bucket:
        _upload_scores_file(output_file_name, output_bucket, output_name)
    print(format_evaluation_timings(timings, output_name))
