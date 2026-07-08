# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Recompute headline viewer numbers from the published artifacts and assert they match.

A published viewer-artifact tree (``insights.json`` / ``datasets.json`` / ``scores-summary.json``
indexing the per-dataset Class-4 match-up parquet, year JSON and eddy census) carries derived
headline numbers whose single source of truth is the match-up parquet. This module re-derives
those numbers straight from the parquet and checks them against the published aggregates, so a
report can be trusted without re-running the whole evaluation:

- **Class-4 pooled RMSD** per ``(variable, depth_bin, lead)`` recomputed as the observation-pooled
  ``sqrt(mean(abs_error ** 2))`` (algebraically the n-weighted recombination the official
  ``class4_rmsd`` aggregate uses) and compared to ``scores-summary.json`` at ``class4_rtol``. When
  the official record carries the pooled observation count ``n`` (emitted by the aggregation
  library), the parquet's finite-obs count is additionally asserted against it — an independent
  guard that catches uniform obs thinning an RMSD tolerance alone would miss. On older
  ``scores-summary.json`` without ``n`` the count assertion is skipped (and logged as skipped, so
  it degrades gracefully rather than silently passing as if checked).
- **Year by-start pooled vs official** — the published ``year-rmsd-by-start.json`` per-start RMSD
  series recombined n-weighted over its starts (``sqrt(sum(rmsd ** 2 * n) / sum(n))``) and compared
  to the official ``class4_rmsd`` in ``scores-summary.json``. This is an *independent* cross-check:
  the year JSON derives from the match-up parquet while the official value derives from the
  separate canonical ``scores.parquet``, so agreement validates the year artifact against the
  official score (not merely its own materialization).
- **Year per-start RMSD/bias (materialization)** recomputed pooled over all match-ups of a sampled
  start and compared to ``year-rmsd-by-start.json`` (a documented sample of starts per variable
  keeps runtime sane). This is a *materialization-consistency* check (parquet <-> derived JSON): it
  proves the JSON was faithfully materialized from the same parquet, not that it agrees with the
  official scores (the by-start-vs-official check above covers that).
- **Year error-geography (materialization)** recomputed as the per-cell mean absolute error and
  compared to ``year-error-geography.json`` for a random sample of cells per variable. Also a
  materialization-consistency check (parquet <-> derived JSON), with no independent official
  counterpart (the official scores carry no per-cell geography).
- **Eddy census** structurally validated (schema of every detection, provenance/parameter block
  presence); no recompute, only integrity.

The single streaming pass over the (large) parquet accumulates every quantity at once. The result
is a machine-readable verification report (per-check pass/fail, numbers, tolerances, timestamp,
library version); :func:`reconcile_viewer_artifacts` raises :class:`ReconciliationError` on any
failure, and the CLI turns that into a non-zero exit.
"""

from collections import defaultdict
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import tempfile
import urllib.request

import numpy
import pandas
import pyarrow.parquet

from oceanbench.core.schema_validation import load_schema
from oceanbench.publish.viewer_artifacts import (
    _YEAR_GEOGRAPHY_DECIMALS,
    _YEAR_TARGETS,
    _cosine_latitude_weights,  # noqa: F401 - kept alongside the year grid helpers for provenance parity
    _grid_cells,
    _year_grid_for_region,
    provenance_block,
)

INSIGHTS_INDEX_FILENAME = "insights.json"
RECONCILIATION_REPORT_FILENAME = "reconciliation-report.json"

DEFAULT_CLASS4_RELATIVE_TOLERANCE = 1e-6
DEFAULT_YEAR_RMSD_RELATIVE_TOLERANCE = 1e-4
DEFAULT_YEAR_STARTS_PER_VARIABLE = 4
DEFAULT_GEOGRAPHY_CELLS_PER_VARIABLE = 20
DEFAULT_SEED = 20260707

# The pooled obs count and the parquet finite-obs count are both integer counts of the same
# match-ups, so an exact match is expected; a sub-unit tolerance passes exact-integer equality
# (including the float representation the JSON carries) while failing any off-by-one drift.
DEFAULT_CLASS4_OBS_COUNT_TOLERANCE = 0.5

_LOGGER = logging.getLogger(__name__)

_CLASS4_METRIC = "class4_rmsd"
_SHORT_TO_TARGET = {short: (variable, depth_bin) for variable, depth_bin, short in _YEAR_TARGETS}


class ReconciliationError(AssertionError):
    """Raised when a recomputed headline number disagrees with its published aggregate."""


@dataclass(frozen=True)
class _Accumulators:
    class4_square_sum: dict
    class4_count: dict
    start_square_sum: dict
    start_count: dict
    start_signed_sum: dict
    cell_absolute_sum: dict
    cell_count: dict


def _is_url(location: str) -> bool:
    return location.startswith("http://") or location.startswith("https://")


def _join(base: str, relative: str) -> str:
    relative = relative[2:] if relative.startswith("./") else relative
    if _is_url(base):
        return base.rstrip("/") + "/" + relative
    return str(Path(base) / relative)


def _content_root(artifacts_base: str) -> str:
    """The base the index-file relative paths resolve against (the parent of the index directory)."""
    if _is_url(artifacts_base):
        return artifacts_base.rstrip("/").rsplit("/", 1)[0]
    return str(Path(artifacts_base).parent)


def _read_json(location: str) -> dict:
    if _is_url(location):
        with urllib.request.urlopen(location) as response:
            return json.loads(response.read())
    return json.loads(Path(location).read_text(encoding="utf-8"))


def _local_parquet_path(location: str, downloads: list[str]) -> str:
    if not _is_url(location):
        return location
    handle = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    handle.close()
    urllib.request.urlretrieve(location, handle.name)
    downloads.append(handle.name)
    return handle.name


def _empty_accumulators() -> _Accumulators:
    return _Accumulators(
        class4_square_sum=defaultdict(float),
        class4_count=defaultdict(int),
        start_square_sum=defaultdict(float),
        start_count=defaultdict(int),
        start_signed_sum=defaultdict(float),
        cell_absolute_sum={},
        cell_count={},
    )


def _accumulate_parquet(parquet_path: str, grid: dict) -> _Accumulators:
    accumulators = _empty_accumulators()
    cell_count = grid["nlat"] * grid["nlon"]
    parquet_file = pyarrow.parquet.ParquetFile(parquet_path)
    columns = [
        "variable",
        "depth_bin",
        "lead_day",
        "start_date",
        "latitude",
        "longitude",
        "observation_value",
        "model_value",
        "abs_error",
    ]
    for group_index in range(parquet_file.num_row_groups):
        batch = parquet_file.read_row_group(group_index, columns=columns)
        _accumulate_batch(batch, accumulators, grid, cell_count)
    _fold_class4_from_starts(accumulators)
    return accumulators


def _accumulate_batch(batch, accumulators: _Accumulators, grid: dict, cell_count: int) -> None:
    """Accumulate one row group, grouping on the actual per-row keys (no purity assumption).

    The served layout intends one ``(start_date, lead_day, variable, depth_bin)`` group per row
    group, but a published parquet may violate that; grouping on the real column values keeps the
    recomputation correct either way.
    """
    variable = numpy.asarray(batch["variable"].to_pylist())
    depth_bin = numpy.asarray(batch["depth_bin"].to_pylist())
    lead_day = batch["lead_day"].to_numpy().astype("int64")
    start_date = numpy.asarray(batch["start_date"].to_pylist())
    latitude = batch["latitude"].to_numpy().astype("float64")
    longitude = batch["longitude"].to_numpy().astype("float64")
    signed = batch["model_value"].to_numpy().astype("float64") - batch["observation_value"].to_numpy().astype("float64")
    absolute = batch["abs_error"].to_numpy().astype("float64")
    finite = numpy.isfinite(absolute) & numpy.isfinite(latitude) & numpy.isfinite(longitude)

    variable, depth_bin = variable[finite], depth_bin[finite]
    lead_day, start_date = lead_day[finite], start_date[finite]
    latitude, longitude = latitude[finite], longitude[finite]
    signed, absolute = signed[finite], absolute[finite]
    square = absolute * absolute
    cell, cell_valid = _grid_cells(latitude, longitude, grid)

    frame = pandas.DataFrame(
        {
            "variable": variable,
            "depth_bin": depth_bin,
            "lead_day": lead_day,
            "start_date": start_date,
            "square": square,
            "signed": signed,
            "absolute": absolute,
            "cell": cell,
            "cell_valid": cell_valid,
        }
    )
    for start_key, group in frame.groupby(["variable", "depth_bin", "lead_day", "start_date"], sort=False):
        variable_name, depth_name, lead_value, start_value = start_key
        key = (variable_name, depth_name, int(lead_value), start_value)
        accumulators.start_square_sum[key] += float(group["square"].sum())
        accumulators.start_count[key] += int(len(group))
        accumulators.start_signed_sum[key] += float(group["signed"].sum())

    for geography_key, group in frame[frame["cell_valid"]].groupby(["variable", "depth_bin", "lead_day"], sort=False):
        variable_name, depth_name, lead_value = geography_key
        key = (variable_name, depth_name, int(lead_value))
        if key not in accumulators.cell_absolute_sum:
            accumulators.cell_absolute_sum[key] = numpy.zeros(cell_count)
            accumulators.cell_count[key] = numpy.zeros(cell_count, dtype=numpy.int64)
        numpy.add.at(accumulators.cell_absolute_sum[key], group["cell"].to_numpy(), group["absolute"].to_numpy())
        numpy.add.at(accumulators.cell_count[key], group["cell"].to_numpy(), 1)


def _fold_class4_from_starts(accumulators: _Accumulators) -> None:
    for (variable, depth_bin, lead_day, _), square_sum in accumulators.start_square_sum.items():
        metric_key = (variable, depth_bin, lead_day)
        accumulators.class4_square_sum[metric_key] += square_sum
    for (variable, depth_bin, lead_day, _), count in accumulators.start_count.items():
        accumulators.class4_count[(variable, depth_bin, lead_day)] += count


def _relative_difference(recomputed: float, official: float) -> float:
    denominator = abs(official)
    if denominator == 0.0:
        return abs(recomputed - official)
    return abs(recomputed - official) / denominator


def _class4_checks(
    accumulators: _Accumulators,
    summary,
    dataset: str,
    region: str,
    relative_tolerance: float,
    obs_count_tolerance: float = DEFAULT_CLASS4_OBS_COUNT_TOLERANCE,
) -> list[dict]:
    official = {
        (record["variable"], record["depth"], record["lead_day"]): record
        for record in summary
        if record.get("metric") == _CLASS4_METRIC
        and record.get("challenger") == dataset
        and record.get("region") == region
    }
    checks = []
    for metric_key, record in sorted(official.items()):
        variable, depth_bin, lead_day = metric_key
        expected = record["mean"]
        official_n = record.get("n")
        count = accumulators.class4_count.get(metric_key, 0)
        if count == 0:
            checks.append(
                {
                    "check": "class4_pooled_rmsd",
                    "key": {"variable": variable, "depth_bin": depth_bin, "lead_day": lead_day},
                    "passed": False,
                    "message": "no match-up rows for an aggregate present in scores-summary.json",
                }
            )
            continue
        recomputed = float(numpy.sqrt(accumulators.class4_square_sum[metric_key] / count))
        difference = _relative_difference(recomputed, expected)
        rmsd_passed = difference <= relative_tolerance

        # Independent obs-count guard: the official pooled n (sum of per-start n) must equal the
        # parquet's finite-obs count. Uniform obs thinning leaves the RMSD ~unchanged but shrinks
        # the count, so this catches a regression the RMSD tolerance alone would pass. Skipped
        # (and logged) when the official record predates the pooled-n emission.
        obs_count_checked = official_n is not None
        if obs_count_checked:
            obs_count_difference = abs(count - float(official_n))
            obs_count_passed = obs_count_difference <= obs_count_tolerance
        else:
            obs_count_difference = None
            obs_count_passed = True
            _LOGGER.info(
                "class4 obs-count guard skipped for %s/%s %s/%s/lead%s: scores-summary.json record "
                "carries no pooled n (older artifact); RMSD checked, count not asserted",
                dataset,
                region,
                variable,
                depth_bin,
                lead_day,
            )

        passed = rmsd_passed and obs_count_passed
        if not rmsd_passed:
            message = "class4 pooled RMSD exceeds relative tolerance"
        elif obs_count_checked and not obs_count_passed:
            message = "class4 pooled obs count disagrees with official n"
        else:
            message = None
        checks.append(
            {
                "check": "class4_pooled_rmsd",
                "key": {"variable": variable, "depth_bin": depth_bin, "lead_day": lead_day},
                "recomputed": recomputed,
                "official": expected,
                "observation_count": count,
                "official_n": official_n,
                "obs_count_checked": obs_count_checked,
                "obs_count_difference": obs_count_difference,
                "obs_count_tolerance": obs_count_tolerance,
                "relative_difference": difference,
                "tolerance": relative_tolerance,
                "passed": passed,
                "message": message,
            }
        )
    return checks


def _official_class4_means(summary, dataset: str, region: str) -> dict:
    """Official ``class4_rmsd`` pooled means keyed by ``(variable, depth_bin, lead_day)``."""
    return {
        (record["variable"], record["depth"], record["lead_day"]): record["mean"]
        for record in summary
        if record.get("metric") == _CLASS4_METRIC
        and record.get("challenger") == dataset
        and record.get("region") == region
    }


def _year_by_start_recombination_checks(
    year_rmsd: dict,
    summary,
    dataset: str,
    region: str,
    relative_tolerance: float,
) -> list[dict]:
    """Independent cross-check: pool ``year-rmsd-by-start`` over starts and compare to official.

    Each variable/lead's published per-start RMSD series is recombined n-weighted over its starts
    (``sqrt(sum(rmsd ** 2 * n) / sum(n))``, the same recombination the official ``class4_rmsd``
    aggregate uses) and compared to the official pooled value in ``scores-summary.json``. Unlike
    :func:`_year_rmsd_checks` this is *not* a parquet<->JSON materialization check: the official
    value comes from the separate canonical ``scores.parquet``, so agreement validates the year
    artifact against the official score. The published per-start RMSD is rounded to six decimals,
    so the pooled value tracks the official one to a few 1e-6 (full-precision it matches to machine
    epsilon); ``relative_tolerance`` (the year-RMSD tolerance) comfortably covers the rounding.
    """
    official = _official_class4_means(summary, dataset, region)
    checks = []
    for short, block in year_rmsd["variables"].items():
        if short not in _SHORT_TO_TARGET:
            continue
        variable, depth_bin = _SHORT_TO_TARGET[short]
        for lead_key, series in block["leads"].items():
            lead_day = int(lead_key)
            rmsd_values = numpy.asarray(series["rmsd"], dtype=float)
            counts = numpy.asarray(series["n"], dtype=float)
            key = (variable, depth_bin, lead_day)
            if rmsd_values.size == 0 or counts.sum() <= 0 or key not in official:
                continue
            pooled = float(numpy.sqrt((rmsd_values**2 * counts).sum() / counts.sum()))
            expected = official[key]
            difference = _relative_difference(pooled, expected)
            passed = difference <= relative_tolerance
            checks.append(
                {
                    "check": "year_by_start_pooled_vs_official",
                    "verifies": "independent: year-rmsd-by-start recombined over starts vs official class4_rmsd",
                    "key": {"variable": short, "lead_day": lead_day},
                    "recomputed": pooled,
                    "official": expected,
                    "starts": int(rmsd_values.size),
                    "observation_count": int(counts.sum()),
                    "relative_difference": difference,
                    "tolerance": relative_tolerance,
                    "passed": passed,
                    "message": (
                        None if passed else "pooled year-rmsd-by-start disagrees with official class4_rmsd"
                    ),
                }
            )
    return checks


def _year_rmsd_checks(
    accumulators: _Accumulators,
    year_rmsd: dict,
    relative_tolerance: float,
    starts_per_variable: int,
    generator: numpy.random.Generator,
) -> list[dict]:
    """Materialization consistency (parquet <-> derived JSON), not official-score validation.

    Re-derives a sampled start's pooled RMSD/bias/n straight from the match-up parquet the year
    JSON was itself materialized from and checks they agree, proving the JSON is a faithful
    materialization of that parquet. It does *not* establish agreement with the official scores;
    :func:`_year_by_start_recombination_checks` covers that independent comparison.
    """
    checks = []
    for short, block in year_rmsd["variables"].items():
        if short not in _SHORT_TO_TARGET:
            continue
        variable, depth_bin = _SHORT_TO_TARGET[short]
        for lead_key, series in block["leads"].items():
            lead_day = int(lead_key)
            dates = series["dates"]
            if not dates:
                continue
            chosen = generator.choice(len(dates), size=min(starts_per_variable, len(dates)), replace=False)
            for position in sorted(int(index) for index in chosen):
                start_date = dates[position]
                key = (variable, depth_bin, lead_day, start_date)
                count = accumulators.start_count.get(key, 0)
                published_rmsd = series["rmsd"][position]
                published_n = series["n"][position]
                published_bias = series["bias"][position]
                if count == 0:
                    checks.append(
                        {
                            "check": "year_rmsd_by_start",
                            "key": {"variable": short, "lead_day": lead_day, "start_date": start_date},
                            "passed": False,
                            "message": "no match-up rows for a published start",
                        }
                    )
                    continue
                recomputed_rmsd = float(numpy.sqrt(accumulators.start_square_sum[key] / count))
                recomputed_bias = float(accumulators.start_signed_sum[key] / count)
                rmsd_difference = _relative_difference(recomputed_rmsd, published_rmsd)
                bias_difference = abs(recomputed_bias - published_bias)
                bias_tolerance = max(relative_tolerance * abs(published_bias), 1e-6)
                passed = (
                    (rmsd_difference <= relative_tolerance)
                    and (count == published_n)
                    and (bias_difference <= bias_tolerance)
                )
                checks.append(
                    {
                        "check": "year_rmsd_by_start",
                        "verifies": "materialization (parquet <-> derived JSON), not official scores",
                        "key": {"variable": short, "lead_day": lead_day, "start_date": start_date},
                        "recomputed_rmsd": recomputed_rmsd,
                        "official_rmsd": published_rmsd,
                        "relative_difference": rmsd_difference,
                        "tolerance": relative_tolerance,
                        "recomputed_n": count,
                        "official_n": published_n,
                        "recomputed_bias": recomputed_bias,
                        "official_bias": published_bias,
                        "bias_difference": bias_difference,
                        "passed": passed,
                        "message": None if passed else "year per-start RMSD/n/bias disagrees with published series",
                    }
                )
    return checks


def _geography_checks(
    accumulators: _Accumulators,
    year_geography: dict,
    cells_per_variable: int,
    generator: numpy.random.Generator,
) -> list[dict]:
    """Materialization consistency (parquet <-> derived JSON) for the per-cell error geography.

    Re-derives a sampled cell's mean absolute error from the same match-up parquet and checks it
    against the published raster. Like :func:`_year_rmsd_checks` this proves faithful
    materialization, not agreement with official scores; the official scores carry no per-cell
    geography, so there is no independent counterpart to compare against.
    """
    checks = []
    for short, block in year_geography["variables"].items():
        if short not in _SHORT_TO_TARGET:
            continue
        variable, depth_bin = _SHORT_TO_TARGET[short]
        decimals = _YEAR_GEOGRAPHY_DECIMALS[short]
        tolerance = 1.5 * 10.0 ** (-decimals)
        candidates = [
            (int(lead_key), index)
            for lead_key, values in block["leads"].items()
            for index, value in enumerate(values)
            if value is not None
        ]
        if not candidates:
            continue
        chosen = generator.choice(len(candidates), size=min(cells_per_variable, len(candidates)), replace=False)
        for pick in chosen:
            lead_day, cell = candidates[int(pick)]
            key = (variable, depth_bin, lead_day)
            counts = accumulators.cell_count.get(key)
            published_value = block["leads"][str(lead_day)][cell]
            if counts is None or counts[cell] == 0:
                checks.append(
                    {
                        "check": "year_error_geography",
                        "key": {"variable": short, "lead_day": lead_day, "cell": cell},
                        "passed": False,
                        "message": "no match-up rows for a published non-null cell",
                    }
                )
                continue
            recomputed_value = round(float(accumulators.cell_absolute_sum[key][cell] / counts[cell]), decimals)
            difference = abs(recomputed_value - published_value)
            passed = difference <= tolerance
            checks.append(
                {
                    "check": "year_error_geography",
                    "verifies": "materialization (parquet <-> derived JSON), not official scores",
                    "key": {"variable": short, "lead_day": lead_day, "cell": cell},
                    "recomputed": recomputed_value,
                    "official": published_value,
                    "difference": difference,
                    "tolerance": tolerance,
                    "recomputed_n": int(counts[cell]),
                    "official_n": None,
                    "passed": passed,
                    "message": None if passed else "year error-geography cell disagrees with published value",
                }
            )
    return checks


def _eddy_census_checks(census: dict, root: str) -> list[dict]:
    eddy_schema = load_schema("eddies")["$defs"]["eddy"]
    checks = []
    required = ("kind", "schema_version", "variable", "dataset", "parameters")
    missing = [key for key in required if key not in census]
    if census.get("kind") != "eddy-census":
        missing.append("kind==eddy-census")
    if not isinstance(census.get("parameters"), dict) or not census.get("parameters"):
        missing.append("parameters (non-empty)")
    frames = census.get("frames")
    if frames is None and "leads" in census:
        frames = [_read_json(_join(root, entry["file"])).get("frame", {}) for entry in census["leads"]]
    if not frames:
        missing.append("frames")

    detection_failures = _validate_detections(frames or [], eddy_schema)
    passed = not missing and not detection_failures
    checks.append(
        {
            "check": "eddy_census_integrity",
            "frames": len(frames or []),
            "missing_fields": missing,
            "invalid_detections": detection_failures,
            "has_provenance": "provenance" in census,
            "passed": passed,
            "message": None if passed else "eddy census failed structural or schema validation",
        }
    )
    return checks


def _validate_detections(frames: list, eddy_schema: dict) -> int:
    import jsonschema

    failures = 0
    for frame in frames:
        for detection in frame.get("detections", []):
            try:
                jsonschema.validate(instance=detection, schema=eddy_schema)
            except jsonschema.ValidationError:
                failures += 1
    return failures


def _dataset_targets(insights: dict, dataset_filter: str | None, region_filter: str | None):
    for dataset, regions in insights.get("datasets", {}).items():
        if dataset_filter is not None and dataset != dataset_filter:
            continue
        for region, artifacts in regions.items():
            if region_filter is not None and region != region_filter:
                continue
            yield dataset, region, artifacts


def reconcile_viewer_artifacts(
    artifacts_base: str,
    *,
    dataset: str | None = None,
    region: str | None = None,
    output_path: str | None = None,
    class4_relative_tolerance: float = DEFAULT_CLASS4_RELATIVE_TOLERANCE,
    class4_obs_count_tolerance: float = DEFAULT_CLASS4_OBS_COUNT_TOLERANCE,
    year_rmsd_relative_tolerance: float = DEFAULT_YEAR_RMSD_RELATIVE_TOLERANCE,
    starts_per_variable: int = DEFAULT_YEAR_STARTS_PER_VARIABLE,
    cells_per_variable: int = DEFAULT_GEOGRAPHY_CELLS_PER_VARIABLE,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Recompute and verify every headline number under a published viewer-artifact tree.

    ``artifacts_base`` is the directory (local path or ``https://`` prefix) that holds
    ``insights.json`` and ``scores-summary.json``. Every ``(dataset, region)`` carrying a Class-4
    match-up parquet is reconciled; ``dataset`` / ``region`` narrow that set. The verification
    report is returned, written to ``output_path`` (or next to a local tree), and a
    :class:`ReconciliationError` is raised if any check failed.
    """
    insights = _read_json(_join(artifacts_base, INSIGHTS_INDEX_FILENAME))
    summary_relative = insights.get("scores_summary", "./scores-summary.json")
    root = _content_root(artifacts_base)
    summary = _read_json(_join(root, summary_relative))

    downloads: list[str] = []
    dataset_reports = []
    try:
        for dataset_name, region_name, artifacts in _dataset_targets(insights, dataset, region):
            if not artifacts.get("class4_matchups"):
                continue
            generator = numpy.random.default_rng(seed)
            grid = _year_grid_for_region(region_name)
            parquet_path = _local_parquet_path(_join(root, artifacts["class4_matchups"]), downloads)
            accumulators = _accumulate_parquet(parquet_path, grid)

            checks = _class4_checks(
                accumulators,
                summary,
                dataset_name,
                region_name,
                class4_relative_tolerance,
                class4_obs_count_tolerance,
            )
            if artifacts.get("year_rmsd_by_start"):
                year_rmsd = _read_json(_join(root, artifacts["year_rmsd_by_start"]))
                checks += _year_by_start_recombination_checks(
                    year_rmsd, summary, dataset_name, region_name, year_rmsd_relative_tolerance
                )
                checks += _year_rmsd_checks(
                    accumulators, year_rmsd, year_rmsd_relative_tolerance, starts_per_variable, generator
                )
            if artifacts.get("year_error_geography"):
                year_geography = _read_json(_join(root, artifacts["year_error_geography"]))
                checks += _geography_checks(accumulators, year_geography, cells_per_variable, generator)
            if artifacts.get("eddies"):
                census = _read_json(_join(root, artifacts["eddies"]))
                checks += _eddy_census_checks(census, root)

            dataset_reports.append(
                {
                    "dataset": dataset_name,
                    "region": region_name,
                    "checks": checks,
                    "checks_total": len(checks),
                    "checks_passed": sum(1 for check in checks if check["passed"]),
                    "passed": all(check["passed"] for check in checks),
                }
            )
    finally:
        for path in downloads:
            Path(path).unlink(missing_ok=True)

    checks_total = sum(entry["checks_total"] for entry in dataset_reports)
    checks_passed = sum(entry["checks_passed"] for entry in dataset_reports)
    report = {
        "kind": "viewer-artifacts-reconciliation",
        "provenance": provenance_block(source=artifacts_base),
        "base": artifacts_base,
        "tolerances": {
            "class4_relative": class4_relative_tolerance,
            "class4_obs_count": class4_obs_count_tolerance,
            "year_rmsd_relative": year_rmsd_relative_tolerance,
        },
        "sampling": {
            "year_starts_per_variable": starts_per_variable,
            "geography_cells_per_variable": cells_per_variable,
            "seed": seed,
        },
        "datasets": dataset_reports,
        "checks_total": checks_total,
        "checks_passed": checks_passed,
        "passed": all(entry["passed"] for entry in dataset_reports) and bool(dataset_reports),
    }

    report_path = output_path
    if report_path is None and not _is_url(artifacts_base):
        report_path = str(Path(artifacts_base) / RECONCILIATION_REPORT_FILENAME)
    if report_path is not None:
        Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
        report["report_path"] = report_path

    _print_summary(report)
    if not report["passed"]:
        raise ReconciliationError(
            f"reconciliation failed: {checks_passed}/{checks_total} checks passed"
            + ("" if dataset_reports else " (no dataset with a match-up parquet was reconciled)")
        )
    return report


def _print_summary(report: dict) -> None:
    print(f"viewer-artifact reconciliation of {report['base']}")
    for entry in report["datasets"]:
        marker = "PASS" if entry["passed"] else "FAIL"
        passed_count, total_count = entry["checks_passed"], entry["checks_total"]
        print(f"  [{marker}] {entry['dataset']}/{entry['region']}: {passed_count}/{total_count} checks")
        for check in entry["checks"]:
            if not check["passed"]:
                print(f"      MISMATCH {check['check']} {check.get('key', '')}: {check.get('message')}")
    overall = "PASS" if report["passed"] else "FAIL"
    print(f"  overall [{overall}]: {report['checks_passed']}/{report['checks_total']} checks passed")
