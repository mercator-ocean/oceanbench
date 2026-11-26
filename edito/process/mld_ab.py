# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
TEST ONLY: time the two mixed layer depth metrics of one challenger file under several variants, each in its own
process on the same node, and compare their tables.

    mld_ab.py CHALLENGER_FILE OUT_DIR RUNS [PROFILED_RUN]

RUNS is a comma list of variant names, run in that order (e.g. A,B,A,B). A variant is a set of environment
variables (VARIANTS below). The run whose index (from 0) is PROFILED_RUN is wrapped in py-spy.
"""

import json
import os
import pickle
import resource
import subprocess
import sys
import time
from pathlib import Path

VARIANTS = {
    # ai-199-opt-main: each metric computes the challenger and the reference mixed layer depth itself.
    "A": {"OCEANBENCH_MLD_REUSE": "0", "OCEANBENCH_MLD_STORE": "", "OCEANBENCH_MLD_FAST": "0"},
    # reuse of the challenger mixed layer depth and the persistent reference store.
    "B": {"OCEANBENCH_MLD_REUSE": "1", "OCEANBENCH_MLD_FAST": "0"},
    # B with the faster mixed layer depth computation.
    "C": {"OCEANBENCH_MLD_REUSE": "1", "OCEANBENCH_MLD_FAST": "1"},
    # A with the faster mixed layer depth computation (checks the computation alone against A).
    "D": {"OCEANBENCH_MLD_REUSE": "0", "OCEANBENCH_MLD_STORE": "", "OCEANBENCH_MLD_FAST": "1"},
}

METRICS = (
    "rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis",
    "rmsd_of_mixed_layer_depth_compared_to_glo12_analysis",
)


def _cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def run_one(challenger_file: str, output_file: str) -> None:
    from oceanbench.core import metrics
    from oceanbench.core.evaluate import open_challenger_dataset

    wall, cpu = time.perf_counter(), _cpu_seconds()
    challenger_dataset = open_challenger_dataset(Path(challenger_file).read_text())
    timings = {"open": (time.perf_counter() - wall, _cpu_seconds() - cpu)}
    tables = {}
    for name in METRICS:
        wall, cpu = time.perf_counter(), _cpu_seconds()
        tables[name] = getattr(metrics, name)(challenger_dataset)
        timings[name] = (time.perf_counter() - wall, _cpu_seconds() - cpu)
        print(f"[mld-ab] {name}: wall {timings[name][0]:.1f} s, cpu {timings[name][1]:.1f} s", flush=True)
    with open(output_file, "wb") as file:
        pickle.dump({"timings": timings, "tables": tables, "maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}, file)


def _compare(first: dict, second: dict) -> dict:
    import numpy

    result = {}
    for name in METRICS:
        a, b = first["tables"][name], second["tables"][name]
        same_shape = a.shape == b.shape and list(a.columns) == list(b.columns) and list(a.index) == list(b.index)
        if not same_shape:
            result[name] = {"same_shape": False}
            continue
        for column_kind, columns in (
            ("missing", [c for c in a.columns if "issing" in str(c)]),
            ("value", [c for c in a.columns if "issing" not in str(c)]),
        ):
            x = numpy.asarray(a[columns].apply(lambda s: s.astype(float, errors="ignore")).values, dtype=float)
            y = numpy.asarray(b[columns].apply(lambda s: s.astype(float, errors="ignore")).values, dtype=float)
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
    challenger_file, out_dir, runs = sys.argv[1], Path(sys.argv[2]), sys.argv[3].split(",")
    profiled = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "" else -1
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for index, variant in enumerate(runs):
        environment = {**os.environ, **VARIANTS[variant]}
        if environment.get("OCEANBENCH_MLD_STORE") == "":
            del environment["OCEANBENCH_MLD_STORE"]
        output = out_dir / f"run{index}-{variant}.pkl"
        command = [sys.executable, "-u", __file__, "--one", challenger_file, str(output)]
        if index == profiled:
            svg, raw = out_dir / f"run{index}-{variant}.svg", out_dir / f"run{index}-{variant}.raw.txt"
            command = ["py-spy", "record", "--native", "--threads", "--rate", "20", "--format", "raw", "-o", str(raw), "--"] + command
        print(f"[mld-ab] === run {index} variant {variant}: {' '.join(command)}", flush=True)
        wall = time.perf_counter()
        children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        code = subprocess.call(command, env=environment)
        children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu = (children_after.ru_utime - children_before.ru_utime) + (children_after.ru_stime - children_before.ru_stime)
        if code != 0 and index == profiled:
            print(f"[mld-ab] profiled run failed ({code}), rerunning without py-spy", flush=True)
            code = subprocess.call(command[command.index("--") + 1 :], env=environment)
        record = {"index": index, "variant": variant, "exit": code, "process_wall": time.perf_counter() - wall, "process_cpu": child_cpu}
        if output.exists():
            with open(output, "rb") as file:
                record.update({k: v for k, v in pickle.load(file).items() if k != "tables"})
        print(f"[mld-ab] RESULT {json.dumps(record)}", flush=True)
        summary.append(record)
    if profiled >= 0:
        raw = out_dir / f"run{profiled}-{runs[profiled]}.raw.txt"
        if raw.exists():
            # Fold the raw stacks into the flame graph too, by self time per frame for the log.
            subprocess.call(["py-spy", "--version"])
            totals: dict[str, int] = {}
            for line in raw.read_text().splitlines():
                stack, _, count = line.rpartition(" ")
                frames = stack.split(";")
                if frames and count.isdigit():
                    totals[frames[-1]] = totals.get(frames[-1], 0) + int(count)
            print("[mld-ab] top self frames:", flush=True)
            for frame, count in sorted(totals.items(), key=lambda item: -item[1])[:40]:
                print(f"[mld-ab]   {count:8d} {frame}", flush=True)
    by_variant: dict[str, dict] = {}
    for index, variant in enumerate(runs):
        path = out_dir / f"run{index}-{variant}.pkl"
        if path.exists():
            with open(path, "rb") as file:
                by_variant.setdefault(variant, pickle.load(file))
    reference = runs[0]
    for variant, loaded in by_variant.items():
        if variant != reference:
            print(f"[mld-ab] IDENTITY {variant} vs {reference}: {json.dumps(_compare(by_variant[reference], loaded))}", flush=True)
    for variant, loaded in by_variant.items():
        for name, table in loaded["tables"].items():
            table.to_csv(out_dir / f"{variant}.{name}.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
