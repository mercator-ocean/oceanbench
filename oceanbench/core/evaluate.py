# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from os import environ
from pathlib import PurePosixPath

from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable

from oceanbench.core.python2jupyter import (
    generate_evaluation_notebook_file,
)
from papermill import execute_notebook


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


def _derive_output_notebook_file_name(challenger_path: str) -> str:
    stem = PurePosixPath(challenger_path).stem
    return f"{stem}.report.ipynb"


def evaluate_challenger(
    challenger_python_code_uri_or_local_path: str | None = None,
    output_bucket: str | None = None,
    output_prefix: str | None = None,
):
    """
    Compute all the benchmark scores for the given challenger dataset, by calling all functions of the `metrics` module.
    It generates and executes a notebook based on the python code that opens the challenger dataset as `challenger_dataset: xarray.Dataset`.

    This function is used for official evaluation.

    The output notebook file name is automatically derived from the challenger file name:
    ``glonet.py`` becomes ``glonet.report.ipynb``.

    Parameters
    ----------
    challenger_python_code_uri_or_local_path : str, optional
        The python content that opens the challenger dataset. Required. Can be a remote file (URL), a DataURI or the path to a local file. Can also be configured with environment variable ``OCEANBENCH_CHALLENGER_PYTHON_CODE_URI_OR_LOCAL_PATH``.
    output_bucket : str, optional
        The destination S3 bucket of the executed notebook. If not provided, the notebook is written on the local filesystem. If provided, uses AWS S3 environment variables. Can also be configured with environment variable ``OCEANBENCH_OUTPUT_BUCKET``.
    output_prefix : str, optional
        The destination S3 prefix of the executed notebook. If ``output_bucket`` is not provided, this option is ignored. If provided, uses AWS S3 environment variables. Can also be configured with environment variable ``OCEANBENCH_OUTPUT_PREFIX``.
    """  # noqa

    oceanbench_challenger_python_code_uri_or_local_path = _parse_input_mandatory(
        challenger_python_code_uri_or_local_path,
        OceanbenchEnvironmentVariable.OCEANBENCH_CHALLENGER_PYTHON_CODE_URI_OR_LOCAL_PATH,
    )
    oceanbench_output_notebook_file_name = _derive_output_notebook_file_name(
        oceanbench_challenger_python_code_uri_or_local_path
    )
    oceanbench_output_bucket = _parse_input_non_manadatory(
        output_bucket,
        OceanbenchEnvironmentVariable.OCEANBENCH_OUTPUT_BUCKET,
    )
    oceanbench_output_prefix = _parse_input_non_manadatory(
        output_prefix,
        OceanbenchEnvironmentVariable.OCEANBENCH_OUTPUT_PREFIX,
    )
    _evaluate_challenger(
        oceanbench_challenger_python_code_uri_or_local_path,
        oceanbench_output_notebook_file_name,
        oceanbench_output_bucket,
        oceanbench_output_prefix,
    )


def _execute_evaluation_notebook_file(
    output_notebook_file_name: str,
    output_bucket: str | None,
    output_prefix: str | None,
):
    environ.setdefault("BOTO3_ENDPOINT_URL", f"https://{environ['AWS_S3_ENDPOINT']}")

    output_name = f"{output_prefix}/{output_notebook_file_name}" if output_prefix else output_notebook_file_name
    output_path = f"s3://{output_bucket}/{output_name}" if output_bucket else output_notebook_file_name
    execute_notebook(
        output_notebook_file_name,
        output_path,
    )


def _evaluate_challenger(
    challenger_python_code_uri_or_local_path: str,
    output_notebook_file_name: str,
    output_bucket: str | None,
    output_prefix: str | None,
):
    generate_evaluation_notebook_file(
        challenger_python_code_uri_or_local_path,
        output_notebook_file_path=output_notebook_file_name,
    )
    _execute_evaluation_notebook_file(
        output_notebook_file_name,
        output_bucket,
        output_prefix,
    )
