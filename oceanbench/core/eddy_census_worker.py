# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Worker process of the eddy census, started by ``eddy_census.detect_eddies_in_fields`` as
``python -m oceanbench.core.eddy_census_worker``.

It never imports the script of the parent process. Each line read from standard input is the path of a
pickled ``(field, date)``; the worker pickles the eddies (or the error) to that path plus ``.out`` and
answers ``ok`` or ``error`` on the descriptor that was standard output when it started. Anything printed
while detecting goes to standard error, so it cannot corrupt the answers. It stops at end of input.
"""

import os
import pickle
import sys
import traceback

RESULT_SUFFIX = ".out"
OK = "ok"
ERROR = "error"


def main() -> None:
    answers = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    from oceanbench.core.eddy_census import detect_eddies_in_field

    for line in sys.stdin:
        task_path = line.rstrip("\n")
        try:
            with open(task_path, "rb") as task_file:
                field, date = pickle.load(task_file)
            result, status = detect_eddies_in_field(field, date), OK
        except Exception as error:
            result, status = f"{type(error).__name__}: {error}\n{traceback.format_exc()}", ERROR
        with open(task_path + RESULT_SUFFIX, "wb") as result_file:
            pickle.dump(result, result_file, protocol=pickle.HIGHEST_PROTOCOL)
        answers.write(status + "\n")


if __name__ == "__main__":
    main()
