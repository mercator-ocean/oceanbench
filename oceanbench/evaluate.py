# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from oceanbench.core.python2jupyter import (
    generate_evaluation_notebook_file,
)


def generate_notebook_to_evaluate(path_to_canditate_python_code_file: str, output_notebook_file_path: str):
    return generate_evaluation_notebook_file(
        python_file_path=path_to_canditate_python_code_file,
        output_notebook_file_path=output_notebook_file_path,
    )
