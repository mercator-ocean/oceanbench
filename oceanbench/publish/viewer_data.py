# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Build the viewer data of one evaluated challenger and region in a directory that mirrors the viewer data prefix.

Besides the viewer files of the challenger, the index files ``datasets.json``, ``insights.json`` and
``scores-summary.json`` are merged with what the directory already holds, so evaluating several
challengers into the same directory gives a complete viewer data tree.

The scores summary reports the official score as its mean. Its 95% interval is a seeded bootstrap
over the start dates of per-start scores: gridded metrics are recomputed one start date at a time
with the ``oceanbench.metrics`` functions, and Class IV per-start scores are pooled from the match-ups
of each start with ``compute_class4_rmsd_table``.
"""

import fcntl
import json
import os
from pathlib import Path

import numpy
import pandas
import xarray

from oceanbench import metrics
from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.regions import RegionSpec, region_to_dict
from oceanbench.core.scores import CLASS4_METRIC_FUNCTION_NAME, METRIC_FUNCTION_NAMES, score_rows
from oceanbench.publish.column_store import COLUMN_STORE_SUFFIX
from oceanbench.publish.viewer_artifacts import (
    BOOTSTRAP_SEED,
    PYRAMID_MANIFEST_SUFFIX,
    PYRAMID_SUFFIX,
    ViewerArtifacts,
    class4_rmsd_rows_by_start,
    confidence_interval,
    write_viewer_artifacts,
)

DATASETS_FILENAME = "datasets.json"
INSIGHTS_FILENAME = "insights.json"
SCORES_SUMMARY_FILENAME = "scores-summary.json"

GRIDDED_METRIC_FUNCTION_NAMES = [
    name for name in METRIC_FUNCTION_NAMES if name.startswith("rmsd_of_") and name != CLASS4_METRIC_FUNCTION_NAME
]
_SUMMARY_REFERENCES = {"glorys_reanalysis": "glorys", "glo12_analysis": "glo12", "observations": "observations"}
_DEPTHLESS_METRICS = {"rmsd_of_mixed_layer_depth", "rmsd_of_geostrophic_currents"}
SUMMARY_BOOTSTRAP_DRAWS = 1000
_CONFIDENCE = 0.95


def _score_key(metric: str, reference: str, standard_name: str, depth: str, lead_day: int) -> tuple:
    return (metric, reference, standard_name, depth, lead_day)


def gridded_scores_by_start(challenger_dataset: xarray.Dataset, region: RegionSpec) -> dict[tuple, numpy.ndarray]:
    """The gridded scores of each start date, from the ``oceanbench.metrics`` functions run on that start alone."""
    start_count = challenger_dataset.sizes[Dimension.FIRST_DAY_DATETIME.key()]
    scores: dict[tuple, numpy.ndarray] = {}
    for start_index in range(start_count):
        start_dataset = challenger_dataset.isel({Dimension.FIRST_DAY_DATETIME.key(): [start_index]})
        for metric_function_name in GRIDDED_METRIC_FUNCTION_NAMES:
            table = getattr(metrics, metric_function_name)(start_dataset, region=region)
            for row in score_rows(metric_function_name, table):
                key = _score_key(row["metric"], row["reference"], row["standard_name"], row["depth"], row["lead_day"])
                value = numpy.nan if row["value"] is None else row["value"]
                scores.setdefault(key, numpy.full(start_count, numpy.nan))[start_index] = value
    return scores


def class4_scores_by_start(
    rows_by_start: pandas.DataFrame, start_dates: list[pandas.Timestamp]
) -> dict[tuple, tuple[numpy.ndarray, numpy.ndarray]]:
    """The pooled Class IV RMSD and scored count of each start date."""
    positions = {start_date: index for index, start_date in enumerate(start_dates)}
    metric, reference = CLASS4_METRIC_FUNCTION_NAME.split("_compared_to_")
    scores = {}
    for (variable, depth_bin, lead_day), group in rows_by_start.groupby(["variable", "depth_bin", "lead_day"]):
        start_positions = [positions[pandas.Timestamp(first_day)] for first_day in group["first_day"]]
        values = numpy.full(len(start_dates), numpy.nan)
        counts = numpy.zeros(len(start_dates))
        values[start_positions] = group["rmsd"].to_numpy(dtype=float)
        counts[start_positions] = (group["count"] - group["missing"]).to_numpy(dtype=float)
        scores[_score_key(metric, reference, variable, depth_bin, int(lead_day) + 1)] = (values, counts)
    return scores


def _mean_over_present_starts(values: numpy.ndarray) -> numpy.ndarray:
    present = numpy.isfinite(values)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        return numpy.where(present, values, 0.0).sum(axis=1) / present.sum(axis=1)


def _pooled_root_mean_square(values: numpy.ndarray, counts: numpy.ndarray) -> numpy.ndarray:
    present = numpy.isfinite(values) & (counts > 0)
    weights = numpy.where(present, counts, 0.0)
    squares = numpy.where(present, values, 0.0) ** 2
    with numpy.errstate(invalid="ignore", divide="ignore"):
        return numpy.sqrt((squares * weights).sum(axis=1) / weights.sum(axis=1))


def _json_value(value: float) -> float | None:
    return None if value is None or not numpy.isfinite(value) else float(value)


def scores_summary_rows(
    challenger_name: str,
    region: RegionSpec,
    document_rows: list[dict],
    start_dates: list[pandas.Timestamp],
    gridded_by_start: dict[tuple, numpy.ndarray],
    class4_by_start: dict[tuple, tuple[numpy.ndarray, numpy.ndarray]],
) -> list[dict]:
    """One summary row per gridded RMSD and Class IV score of the scores file, with its bootstrap interval."""
    bootstrap_indices = numpy.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(start_dates), size=(SUMMARY_BOOTSTRAP_DRAWS, len(start_dates))
    )
    no_values = numpy.full(len(start_dates), numpy.nan)
    rows = []
    for row in document_rows:
        if not row["metric"].startswith("rmsd_of_"):
            continue
        key = _score_key(row["metric"], row["reference"], row["standard_name"], row["depth"], row["lead_day"])
        is_class4 = row["reference"] == "observations"
        if is_class4:
            values, counts = class4_by_start.get(key, (no_values, numpy.zeros(len(start_dates))))
            present = numpy.isfinite(values) & (counts > 0)
            bootstrap = _pooled_root_mean_square(values[bootstrap_indices], counts[bootstrap_indices])
        else:
            values = gridded_by_start.get(key, no_values)
            present = numpy.isfinite(values)
            bootstrap = _mean_over_present_starts(values[bootstrap_indices])
        ci_low, ci_high = confidence_interval(bootstrap, _CONFIDENCE)
        rows.append(
            {
                "challenger": challenger_name,
                "year": start_dates[0].year,
                "region": region.id,
                "metric": "class4_rmsd" if is_class4 else "rmsd",
                "reference": _SUMMARY_REFERENCES[row["reference"]],
                "variable": row["standard_name"],
                "depth": None if row["metric"] in _DEPTHLESS_METRICS else row["depth"],
                "lead_day": row["lead_day"],
                "band": None,
                "polarity": None,
                "unit": row["unit"],
                "mean": row["value"],
                "ci_low": _json_value(ci_low) if row["value"] is not None else None,
                "ci_high": _json_value(ci_high) if row["value"] is not None else None,
                "n_starts": int(present.sum()),
                "skill_baseline": None,
                "skill_vs_persistence_1_degree": None,
                "skill_ci_low": None,
                "skill_ci_high": None,
                "n_starts_paired": None,
                "n": int(counts[present].sum()) if is_class4 else None,
            }
        )
    return rows


def _summary_sort_key(row: dict) -> tuple:
    return (
        row["challenger"],
        row["year"],
        row["region"],
        row["metric"],
        row["reference"],
        row["variable"],
        row["depth"] is None,
        row["depth"] or "",
        row["lead_day"],
    )


def _read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write_index(document, path: Path) -> None:
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def update_viewer_indexes(
    directory: Path,
    dataset_slug: str,
    region: RegionSpec,
    artifacts: ViewerArtifacts,
    summary_rows: list[dict],
) -> None:
    """Merge the entries of one challenger and region into the index files, replacing its previous entries."""
    # Several challengers can be evaluated in parallel processes into the same directory.
    directory_descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(directory_descriptor, fcntl.LOCK_EX)
        if artifacts.has_maps:
            datasets_path = directory / DATASETS_FILENAME
            entries = {entry["slug"]: entry for entry in _read_json(datasets_path, {"datasets": []})["datasets"]}
            entries[dataset_slug] = {
                "slug": dataset_slug,
                "label": entries.get(dataset_slug, {}).get("label", dataset_slug),
                "store": f"./data/{dataset_slug}{PYRAMID_SUFFIX}",
                "manifest": f"./data/{dataset_slug}{PYRAMID_MANIFEST_SUFFIX}",
                "columns": f"./data/{dataset_slug}{COLUMN_STORE_SUFFIX}",
            }
            _write_index({"datasets": [entries[slug] for slug in sorted(entries)]}, datasets_path)

        insights_path = directory / INSIGHTS_FILENAME
        insights = _read_json(insights_path, {"datasets": {}, "regions": {}})
        insights["datasets"].setdefault(dataset_slug, {})[region.id] = artifacts.insights
        insights["regions"][region.id] = region_to_dict(region)["bounds"]
        insights["scores_summary"] = f"./data/{SCORES_SUMMARY_FILENAME}"
        _write_index(insights, insights_path)

        summary_path = directory / SCORES_SUMMARY_FILENAME
        kept_rows = [
            row
            for row in _read_json(summary_path, [])
            if (row["challenger"], row["region"]) != (dataset_slug, region.id)
        ]
        summary_path.write_text(json.dumps(sorted(kept_rows + summary_rows, key=_summary_sort_key)), encoding="utf-8")
    finally:
        os.close(directory_descriptor)


def build_viewer_data(
    directory: str,
    challenger_name: str,
    challenger_dataset: xarray.Dataset,
    region: RegionSpec,
    matchups: pandas.DataFrame | None,
    document_rows: list[dict],
) -> None:
    """
    Write the viewer data of one evaluated challenger and region into ``directory``.

    ``matchups`` are the Class IV match-ups the scores were computed from, ``None`` when the
    observations are unavailable, and ``document_rows`` the rows of the scores file.
    """
    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    has_matchups = matchups is not None and len(matchups) > 0
    rows_by_start = class4_rmsd_rows_by_start(matchups) if has_matchups else None
    artifacts = write_viewer_artifacts(
        output_directory, challenger_name, challenger_dataset, region, matchups, rows_by_start
    )
    start_dates = [
        pandas.Timestamp(start_date) for start_date in challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    ]
    summary_rows = scores_summary_rows(
        challenger_name,
        region,
        document_rows,
        start_dates,
        gridded_scores_by_start(challenger_dataset, region),
        class4_scores_by_start(rows_by_start, start_dates) if has_matchups else {},
    )
    update_viewer_indexes(output_directory, challenger_name, region, artifacts, summary_rows)
