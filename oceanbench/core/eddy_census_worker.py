# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Worker process of the eddy census, started by ``eddy_census.detect_eddies_in_fields`` as
``python -m oceanbench.core.eddy_census_worker``.

It never imports the script of the parent process. Each line read from standard input is the path of a
pickled ``(field, date)``; the worker pickles the eddies (or the error) to that path plus ``.out`` and
answers ``ok`` or ``error`` on the descriptor that was standard output when it started. Anything printed
while detecting goes to standard error, so it cannot corrupt the answers. It frees the memory of each field
before answering and stops at end of input.
"""

import ctypes
import ctypes.util
import gc
import os
import pickle
import sys
import traceback

RESULT_SUFFIX = ".out"
OK = "ok"
ERROR = "error"


def release_memory() -> None:
    """
    Free what the last field left behind: py-eddy-tracker's identification leaves about 1 GB in reference
    cycles and in heap pages the C library keeps, so a worker would otherwise stay near its peak between
    fields and creep up from field to field. Detection results do not depend on it.
    """
    gc.collect()
    library_name = ctypes.util.find_library("c")
    if library_name is None:
        return
    try:
        malloc_trim = ctypes.CDLL(library_name).malloc_trim
    except (OSError, AttributeError):
        # Not glibc (macOS, musl): nothing to trim.
        return
    malloc_trim(0)


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
        field = result = None
        release_memory()
        answers.write(status + "\n")


if __name__ == "__main__":
    main()
