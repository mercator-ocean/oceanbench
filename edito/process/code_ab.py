# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
TEST ONLY: time every metric of one challenger file under two versions of the code, each in its own process on
the same node, and compare their tables.

    code_ab.py CHALLENGER_FILE OUT_DIR RUNS BASE_DIRECTORY

RUNS is a comma list of A and B, run in that order (e.g. A,B,A,B). A runs the oceanbench package found in
BASE_DIRECTORY (put first on PYTHONPATH), B the installed one.
"""

import json
import os
import pickle
import resource
import subprocess
import sys
import time
from pathlib import Path


def _cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def run_one(challenger_file: str, output_file: str) -> None:
    import oceanbench
    from oceanbench.core import metrics
    from oceanbench.core.evaluate import open_challenger_dataset, read_challenger_python_code
    from oceanbench.core.scores import METRIC_FUNCTION_NAMES

    print(f"[code-ab] oceanbench from {oceanbench.__file__}", flush=True)
    wall, cpu = time.perf_counter(), _cpu_seconds()
    challenger_dataset = open_challenger_dataset(read_challenger_python_code(challenger_file))
    timings = {"open": (time.perf_counter() - wall, _cpu_seconds() - cpu)}
    tables = {}
    for name in METRIC_FUNCTION_NAMES:
        wall, cpu = time.perf_counter(), _cpu_seconds()
        tables[name] = getattr(metrics, name)(challenger_dataset)
        timings[name] = (time.perf_counter() - wall, _cpu_seconds() - cpu)
        print(f"[code-ab] {name}: wall {timings[name][0]:.1f} s, cpu {timings[name][1]:.1f} s", flush=True)
    with open(output_file, "wb") as file:
        pickle.dump(
            {"timings": timings, "tables": tables, "maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
            file,
        )


def _numbers(table, columns):
    import numpy

    return numpy.asarray(table[columns].apply(lambda s: s.astype(float, errors="ignore")).values, dtype=float)


def _compare(first: dict, second: dict) -> dict:
    import numpy

    result = {}
    for name in first["tables"]:
        a, b = first["tables"][name], second["tables"].get(name)
        if b is None:
            result[name] = {"missing": True}
            continue
        same_shape = a.shape == b.shape and list(a.columns) == list(b.columns) and list(a.index) == list(b.index)
        if not same_shape:
            result[name] = {"same_shape": False}
            continue
        for column_kind, columns in (
            ("missing", [c for c in a.columns if "issing" in str(c)]),
            ("value", [c for c in a.columns if "issing" not in str(c)]),
        ):
            x, y = _numbers(a, columns), _numbers(b, columns)
            both_nan = numpy.isnan(x) & numpy.isnan(y)
            difference = numpy.where(both_nan, 0.0, numpy.abs(x - y))
            relative = numpy.where(both_nan, 0.0, difference / numpy.maximum(numpy.abs(x), 1e-300))
            result[f"{name} {column_kind}"] = {
                "numbers": int(x.size),
                "differ": int(numpy.count_nonzero(difference)),
                "max_abs": float(numpy.nanmax(difference)) if x.size else 0.0,
                "max_rel": float(numpy.nanmax(relative)) if x.size else 0.0,
                "nan_mismatch": int(numpy.count_nonzero(numpy.isnan(x) != numpy.isnan(y))),
            }
    return result


def main() -> None:
    if sys.argv[1] == "--one":
        run_one(sys.argv[2], sys.argv[3])
        return
    challenger_file, out_dir, runs, base_directory = sys.argv[1], Path(sys.argv[2]), sys.argv[3].split(","), sys.argv[4]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for index, variant in enumerate(runs):
        environment = dict(os.environ)
        if variant == "A":
            environment["PYTHONPATH"] = base_directory
        else:
            environment.pop("PYTHONPATH", None)
        output = out_dir / f"run{index}-{variant}.pkl"
        command = [sys.executable, "-u", __file__, "--one", challenger_file, str(output)]
        print(f"[code-ab] === run {index} variant {variant}", flush=True)
        wall = time.perf_counter()
        code = subprocess.call(command, env=environment, cwd=out_dir)
        record = {"index": index, "variant": variant, "exit": code, "process_wall": time.perf_counter() - wall}
        if output.exists():
            with open(output, "rb") as file:
                record.update({k: v for k, v in pickle.load(file).items() if k != "tables"})
        print(f"[code-ab] RESULT {json.dumps(record)}", flush=True)
        summary.append(record)
    by_variant: dict[str, dict] = {}
    for index, variant in enumerate(runs):
        path = out_dir / f"run{index}-{variant}.pkl"
        if path.exists():
            with open(path, "rb") as file:
                by_variant.setdefault(variant, pickle.load(file))
    if "A" in by_variant and "B" in by_variant:
        print(f"[code-ab] IDENTITY B vs A: {json.dumps(_compare(by_variant['A'], by_variant['B']))}", flush=True)
    for variant, loaded in by_variant.items():
        for name, table in loaded["tables"].items():
            table.to_csv(out_dir / f"{variant}.{name}.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
