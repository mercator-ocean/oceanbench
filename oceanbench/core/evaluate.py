# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from os import environ
from oceanbench.core.python2jupyter import (
    generate_evaluation_notebook_file,
)
from papermill import execute_notebook


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


def evaluate_challenger(
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
