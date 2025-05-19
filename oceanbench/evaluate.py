# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from enum import Enum
from os import environ

from oceanbench.core import evaluate


class OceanbenchEnvironmentVariable(Enum):
    OCEANBENCH_CHALLENGER_PYTHON_CODE_URI_OR_LOCAL_PATH = "OCEANBENCH_CHALLENGER_PYTHON_CODE_URI_OR_LOCAL_PATH"
    OCEANBENCH_OUTPUT_NOTEBOOK_FILE_NAME = "OCEANBENCH_OUTPUT_NOTEBOOK_FILE_NAME"
    OCEANBENCH_OUTPUT_BUCKET = "OCEANBENCH_OUTPUT_BUCKET"
    OCEANBENCH_OUTPUT_PREFIX = "OCEANBENCH_OUTPUT_PREFIX"


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


def evaluate_challenger(
    challenger_python_code_uri_or_local_path: str | None = None,
    output_notebook_file_name: str | None = None,
    output_bucket: str | None = None,
    output_prefix: str | None = None,
):
    oceanbench_challenger_python_code_uri_or_local_path = _parse_input_mandatory(
        challenger_python_code_uri_or_local_path,
        OceanbenchEnvironmentVariable.OCEANBENCH_CHALLENGER_PYTHON_CODE_URI_OR_LOCAL_PATH,
    )
    oceanbench_output_notebook_file_name = _parse_input_mandatory(
        output_notebook_file_name,
        OceanbenchEnvironmentVariable.OCEANBENCH_OUTPUT_NOTEBOOK_FILE_NAME,
    )
    oceanbench_output_bucket = _parse_input_non_manadatory(
        output_bucket,
        OceanbenchEnvironmentVariable.OCEANBENCH_OUTPUT_BUCKET,
    )
    oceanbench_output_prefix = _parse_input_non_manadatory(
        output_prefix,
        OceanbenchEnvironmentVariable.OCEANBENCH_OUTPUT_PREFIX,
    )
    evaluate.evaluate_challenger(
        oceanbench_challenger_python_code_uri_or_local_path,
        oceanbench_output_notebook_file_name,
        oceanbench_output_bucket,
        oceanbench_output_prefix,
    )
