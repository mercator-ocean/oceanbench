# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import PurePosixPath

from oceanbench.core.runtime_configuration import RuntimeConfiguration
from oceanbench.core.python2jupyter import generate_evaluation_notebook_file
from papermill import execute_notebook


def _derive_output_notebook_file_name(challenger_path: str) -> str:
    stem = PurePosixPath(challenger_path).stem
    return f"{stem}.report.ipynb"


def evaluate_challenger(
    challenger_python_code_uri_or_local_path: str,
    output_bucket: str | None = None,
    output_prefix: str | None = None,
    runtime_configuration: RuntimeConfiguration | None = None,
):
    """
    Compute all the benchmark scores for the given challenger dataset, by calling all functions of the `metrics` module.
    It generates and executes a notebook based on the python code that opens the challenger dataset as `challenger_dataset: xarray.Dataset`.

    This function is used for official evaluation.

    The output notebook file name is automatically derived from the challenger file name:
    ``glonet.py`` becomes ``glonet.report.ipynb``.

    Parameters
    ----------
    challenger_python_code_uri_or_local_path : str
        The python content that opens the challenger dataset. Required. Can be a remote file (URL), a DataURI or the path to a local file.
    output_bucket : str, optional
        The destination S3 bucket of the executed notebook. If not provided, the notebook is written on the local filesystem. If provided, uses AWS S3 environment variables.
    output_prefix : str, optional
        The destination S3 prefix of the executed notebook. If ``output_bucket`` is not provided, this option is ignored. If provided, uses AWS S3 environment variables.
    runtime_configuration : RuntimeConfiguration, optional
        Runtime settings applied inside the generated notebook execution, including staging and remote retry behavior.
    """  # noqa
    if challenger_python_code_uri_or_local_path in (None, ""):
        raise ValueError("challenger_python_code_uri_or_local_path is required.")
    resolved_runtime_configuration = runtime_configuration or RuntimeConfiguration()
    output_notebook_file_name = _derive_output_notebook_file_name(challenger_python_code_uri_or_local_path)
    _evaluate_challenger(
        challenger_python_code_uri_or_local_path,
        output_notebook_file_name,
        output_bucket,
        output_prefix,
        resolved_runtime_configuration,
    )


def _execute_evaluation_notebook_file(
    output_notebook_file_name: str,
    output_bucket: str | None,
    output_prefix: str | None,
):
    output_name = f"{output_prefix}/{output_notebook_file_name}" if output_prefix else output_notebook_file_name
    if output_bucket:
        from os import environ

        environ.setdefault("BOTO3_ENDPOINT_URL", f"https://{environ['AWS_S3_ENDPOINT']}")
        output_path = f"s3://{output_bucket}/{output_name}"
    else:
        output_path = output_notebook_file_name
    execute_notebook(
        output_notebook_file_name,
        output_path,
    )


def _evaluate_challenger(
    challenger_python_code_uri_or_local_path: str,
    output_notebook_file_name: str,
    output_bucket: str | None,
    output_prefix: str | None,
    runtime_configuration: RuntimeConfiguration,
):
    generate_evaluation_notebook_file(
        challenger_python_code_uri_or_local_path,
        output_notebook_file_path=output_notebook_file_name,
        runtime_configuration=runtime_configuration,
    )
    _execute_evaluation_notebook_file(
        output_notebook_file_name,
        output_bucket,
        output_prefix,
    )
