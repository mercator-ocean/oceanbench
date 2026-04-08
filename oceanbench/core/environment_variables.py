# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from enum import Enum


class OceanbenchEnvironmentVariable(Enum):
    OCEANBENCH_CHALLENGER_PYTHON_CODE_URI_OR_LOCAL_PATH = "OCEANBENCH_CHALLENGER_PYTHON_CODE_URI_OR_LOCAL_PATH"
    OCEANBENCH_OUTPUT_BUCKET = "OCEANBENCH_OUTPUT_BUCKET"
    OCEANBENCH_OUTPUT_PREFIX = "OCEANBENCH_OUTPUT_PREFIX"
