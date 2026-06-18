# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Compare OceanBench results computed purely online against the local-cache modes.

Runs a small slice of one challenger three ways -- online, cold local cache, warm local
cache -- and checks the results are identical, reporting the wall-clock time of each.

The point is to confirm the ``OCEANBENCH_LOCAL_CACHE`` toggle changes only speed, never
the numbers, on a cheap slice before committing to a full evaluation. Identical bytes in
mean identical scores out, so this small slice generalises to the whole run.

Examples
--------
Fast read-equivalence check (default, no scoring)::

    python helper_scripts/compare_online_versus_local_cache.py --challenger glonet --weeks 1

Full metric-equivalence check on a region::

    python helper_scripts/compare_online_versus_local_cache.py --challenger glonet --weeks 1 \
        --score --region ibi
"""

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy

from oceanbench import metrics
from oceanbench.core.runtime_configuration import RuntimeConfiguration, set_runtime_configuration
from oceanbench.datasets import challenger as challenger_datasets


def _activate_mode(cache_directory: Path | None, remote_retries: int) -> None:
    set_runtime_configuration(
        RuntimeConfiguration(
            local_cache_directory_path=str(cache_directory) if cache_directory is not None else None,
            remote_retries=remote_retries,
        )
    )


def _challenger_slice(challenger_name: str, weeks: int):
    challenger_dataset = getattr(challenger_datasets, challenger_name)()
    return challenger_dataset.isel(first_day_datetime=slice(0, weeks))


def _small_read_sample(challenger_name: str, weeks: int) -> numpy.ndarray:
    challenger_dataset = _challenger_slice(challenger_name, weeks)
    sampled_variable = sorted(challenger_dataset.data_vars)[0]
    sampled_array = challenger_dataset[sampled_variable]
    if "depth" in sampled_array.dims:
        sampled_array = sampled_array.isel(depth=0)
    sampled_array = sampled_array.sel(latitude=slice(0, 10), longitude=slice(-10, 0))
    return sampled_array.compute().values


def _metric_scores(challenger_name: str, weeks: int, region: str) -> numpy.ndarray:
    challenger_dataset = _challenger_slice(challenger_name, weeks)
    scores = metrics.rmsd_of_variables_compared_to_glorys_reanalysis(challenger_dataset, region=region)
    return scores.to_numpy()


def _result_for_mode(challenger_name: str, weeks: int, region: str, compute_score: bool) -> numpy.ndarray:
    if compute_score:
        return _metric_scores(challenger_name, weeks, region)
    return _small_read_sample(challenger_name, weeks)


def _results_are_identical(reference_result: numpy.ndarray, other_result: numpy.ndarray) -> bool:
    return numpy.allclose(reference_result, other_result, rtol=1e-12, atol=1e-12, equal_nan=True)


def compare(challenger_name: str, weeks: int, region: str, compute_score: bool, remote_retries: int) -> int:
    cache_directory = Path(tempfile.mkdtemp(prefix="oceanbench-cache-compare-"))
    modes = (("online", None), ("cold-cache", cache_directory), ("warm-cache", cache_directory))

    print(f"challenger={challenger_name} weeks={weeks} region={region} score={compute_score}")
    print(f"{'mode':12s}{'seconds':>12s}")
    results: dict[str, numpy.ndarray] = {}
    for mode_label, mode_cache_directory in modes:
        if mode_label == "cold-cache":
            shutil.rmtree(cache_directory, ignore_errors=True)
        _activate_mode(mode_cache_directory, remote_retries)
        started_at = time.perf_counter()
        results[mode_label] = _result_for_mode(challenger_name, weeks, region, compute_score)
        print(f"{mode_label:12s}{time.perf_counter() - started_at:12.2f}")

    shutil.rmtree(cache_directory, ignore_errors=True)
    all_identical = all(
        _results_are_identical(results["online"], results[mode]) for mode in ("cold-cache", "warm-cache")
    )
    print("identical across online, cold-cache, warm-cache:", all_identical)
    return 0 if all_identical else 1


def _parse_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--challenger", default="glonet")
    argument_parser.add_argument("--weeks", type=int, default=1)
    argument_parser.add_argument("--region", default="global")
    argument_parser.add_argument("--score", action="store_true", help="Compare a full RMSD metric instead of a read.")
    argument_parser.add_argument("--remote-retries", type=int, default=5)
    return argument_parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    raise SystemExit(
        compare(
            challenger_name=arguments.challenger,
            weeks=arguments.weeks,
            region=arguments.region,
            compute_score=arguments.score,
            remote_retries=arguments.remote_retries,
        )
    )
