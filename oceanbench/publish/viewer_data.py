# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Build the viewer data of one evaluated challenger and region in a directory that mirrors the viewer data prefix.

Besides the viewer files of the challenger, the index files ``datasets.json``, ``insights.json`` and
``scores-summary.json`` are merged with what the directory already holds, so evaluating several
challengers into the same directory gives a complete viewer data tree.

The scores summary reports the official score as its mean. Its 95% interval is a seeded bootstrap
over the start dates of per-start scores. Gridded per-start scores are the ones the scoring run kept,
or else are recomputed one start date at a time with the ``oceanbench.metrics`` functions. Class IV
per-start scores are pooled from the match-ups of each start with ``compute_class4_rmsd_table``.
"""

import fcntl
import json
import os
from pathlib import Path

import numpy
import pandas
import xarray

from oceanbench import metrics
from oceanbench.core.classIV_matchups import Class4Matchups, class4_matchups
from oceanbench.core.dataset_source import DEFAULT_CHALLENGER_KIND, get_challenger_kind, get_challenger_own_reference
from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.regions import RegionSpec, region_to_dict
from oceanbench.core.scores import CLASS4_METRIC_FUNCTION_NAME, GRIDDED_METRIC_FUNCTION_NAMES, score_rows
from oceanbench.publish.column_store import COLUMN_STORE_SUFFIX
from oceanbench.publish.viewer_artifacts import (
    BOOTSTRAP_SEED,
    PYRAMID_MANIFEST_SUFFIX,
    PYRAMID_SUFFIX,
    ViewerArtifacts,
    class4_rmsd_rows_by_start,
    confidence_interval,
    write_region_eddy_census,
    write_viewer_artifacts,
)

DATASETS_FILENAME = "datasets.json"
INSIGHTS_FILENAME = "insights.json"
SCORES_SUMMARY_FILENAME = "scores-summary.json"
SCORES_BY_START_FILENAME = "scores-by-start.json"

_SUMMARY_REFERENCES = {"glorys_reanalysis": "glorys", "glo12_analysis": "glo12", "observations": "observations"}
_DEPTHLESS_METRICS = {"rmsd_of_mixed_layer_depth", "rmsd_of_geostrophic_currents"}
SUMMARY_BOOTSTRAP_DRAWS = 1000
_CONFIDENCE = 0.95


def _score_key(metric: str, reference: str, standard_name: str, depth: str, lead_day: int) -> tuple:
    return (metric, reference, standard_name, depth, lead_day)


def gridded_scores_from_start_tables(start_tables: dict[str, list[pandas.DataFrame]]) -> dict[tuple, numpy.ndarray]:
    """The gridded scores of each start date, from the score table of each start date of each metric."""
    scores: dict[tuple, numpy.ndarray] = {}
    for metric_function_name, tables in start_tables.items():
        for start_index, table in enumerate(tables):
            for row in score_rows(metric_function_name, table):
                key = _score_key(row["metric"], row["reference"], row["standard_name"], row["depth"], row["lead_day"])
                value = numpy.nan if row["value"] is None else row["value"]
                scores.setdefault(key, numpy.full(len(tables), numpy.nan))[start_index] = value
    return scores


def gridded_scores_by_start(challenger_dataset: xarray.Dataset, region: RegionSpec) -> dict[tuple, numpy.ndarray]:
    """The gridded scores of each start date, from the ``oceanbench.metrics`` functions run on that start alone."""
    start_count = challenger_dataset.sizes[Dimension.FIRST_DAY_DATETIME.key()]
    return gridded_scores_from_start_tables(
        {
            metric_function_name: [
                getattr(metrics, metric_function_name)(
                    challenger_dataset.isel({Dimension.FIRST_DAY_DATETIME.key(): [start_index]}), region=region
                )
                for start_index in range(start_count)
            ]
            for metric_function_name in GRIDDED_METRIC_FUNCTION_NAMES
        }
    )


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


def class4_bias_by_start(matchups: pandas.DataFrame | Class4Matchups) -> pandas.Series:
    """Pooled mean of model minus observation per start date, variable, depth bin and lead day (0-based)."""
    parts = class4_matchups(matchups)
    biases = []
    for first_day in parts.first_days():
        part = parts.start(first_day)
        scored = part.dropna(subset=["observation_value", "model_value"])
        bias = (
            (scored["model_value"] - scored["observation_value"])
            .groupby([scored["variable"], scored["depth_bin"], scored["lead_day"]])
            .mean()
        )
        biases.append(pandas.concat({pandas.Timestamp(first_day): bias}, names=["first_day"]))
    return pandas.concat(biases) if biases else pandas.Series(dtype=float)


def _per_lead(values_by_lead: dict[int, numpy.ndarray], lead_days: list[int], start_count: int) -> list[list]:
    return [
        [_json_value(value) for value in values_by_lead.get(lead_day, numpy.full(start_count, numpy.nan))]
        for lead_day in lead_days
    ]


def scores_by_start_document(
    challenger_name: str,
    region_id: str,
    start_dates: list[pandas.Timestamp],
    gridded_by_start: dict[tuple, numpy.ndarray] | None,
    class4_by_start: dict[tuple, tuple[numpy.ndarray, numpy.ndarray]] | None,
    class4_bias: dict[tuple, numpy.ndarray] | None,
    dropped_reference: str | None = None,
    gridded_unavailable_reason: str | None = None,
) -> dict:
    """
    Per-start scores for period-restricted curves, as arrays over ``dates`` (one array per lead day).

    ``gridded`` series hold the RMSD of each start (the year score is their mean over present starts).
    ``class4`` series hold the pooled RMSD, bias and scored count ``n`` of each start (the year score is
    ``sqrt(sum n rmsd^2 / sum n)``). Values are unrounded; ``null`` marks a missing start.
    """
    start_count = len(start_dates)
    lead_days = sorted(
        {key[4] for key in (gridded_by_start or {})} | {key[4] for key in (class4_by_start or {})}
    ) or list(range(1, 11))
    document: dict = {
        "schema_version": 1,
        "challenger": challenger_name,
        "region": region_id,
        "dates": [start_date.strftime("%Y-%m-%d") for start_date in start_dates],
        "lead_days": lead_days,
        "layout": "values[lead_index][start_index]",
        "gridded": {
            "available": gridded_by_start is not None,
            "aggregation": "arithmetic mean over present starts of the per-start RMSD",
            "series": [],
        },
        "class4": {
            "available": class4_by_start is not None,
            "aggregation": "rmsd: sqrt(sum n rmsd^2 / sum n); bias: sum n bias / sum n",
            "series": [],
        },
    }
    if gridded_by_start is None:
        document["gridded"]["reason"] = gridded_unavailable_reason or "per-start gridded scores unavailable"
    else:
        grouped: dict[tuple, dict[int, numpy.ndarray]] = {}
        for (metric, reference, variable, depth, lead_day), values in gridded_by_start.items():
            if reference == "observations":
                continue
            reference_label = _SUMMARY_REFERENCES[reference]
            if reference_label == dropped_reference:
                continue
            grouped.setdefault((metric, reference_label, variable, depth), {})[lead_day] = values
        for (metric, reference, variable, depth), by_lead in sorted(grouped.items(), key=lambda item: str(item[0])):
            document["gridded"]["series"].append(
                {
                    "metric": "rmsd",
                    "function": metric,
                    "reference": reference,
                    "variable": variable,
                    "depth": None if metric in _DEPTHLESS_METRICS else depth,
                    "rmsd": _per_lead(by_lead, lead_days, start_count),
                }
            )
    if class4_by_start is not None:
        grouped4: dict[tuple, dict[int, tuple]] = {}
        for (metric, reference, variable, depth_bin, lead_day), (values, counts) in class4_by_start.items():
            bias = (class4_bias or {}).get((variable, depth_bin, lead_day), numpy.full(start_count, numpy.nan))
            grouped4.setdefault((variable, depth_bin), {})[lead_day] = (values, counts, bias)
        for (variable, depth_bin), by_lead in sorted(grouped4.items()):
            document["class4"]["series"].append(
                {
                    "metric": "class4_rmsd",
                    "reference": "observations",
                    "variable": variable,
                    "depth": depth_bin,
                    "rmsd": _per_lead({k: v[0] for k, v in by_lead.items()}, lead_days, start_count),
                    "bias": _per_lead({k: v[2] for k, v in by_lead.items()}, lead_days, start_count),
                    "n": [
                        [int(count) for count in by_lead[lead_day][1]] if lead_day in by_lead else [0] * start_count
                        for lead_day in lead_days
                    ],
                }
            )
    return document


def class4_bias_arrays(bias: pandas.Series, start_dates: list[pandas.Timestamp]) -> dict[tuple, numpy.ndarray]:
    """Per-start bias arrays keyed (variable, depth_bin, lead_day 1-based), aligned on ``start_dates``."""
    positions = {start_date: index for index, start_date in enumerate(start_dates)}
    arrays: dict[tuple, numpy.ndarray] = {}
    for (first_day, variable, depth_bin, lead_day), value in bias.items():
        key = (variable, depth_bin, int(lead_day) + 1)
        arrays.setdefault(key, numpy.full(len(start_dates), numpy.nan))[positions[pandas.Timestamp(first_day)]] = value
    return arrays


def write_scores_by_start(directory: Path, document: dict) -> str:
    """Write ``insights/<slug>/<region>/scores-by-start.json`` and return its viewer path."""
    insights_directory = directory / "insights" / document["challenger"] / document["region"]
    insights_directory.mkdir(parents=True, exist_ok=True)
    (insights_directory / SCORES_BY_START_FILENAME).write_text(json.dumps(document), encoding="utf-8")
    return f"./data/insights/{document['challenger']}/{document['region']}/{SCORES_BY_START_FILENAME}"


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


def _resolution_label(cell_size_degrees: float) -> str:
    cells_per_degree = round(1 / cell_size_degrees)
    if cells_per_degree >= 1 and abs(cells_per_degree * cell_size_degrees - 1) < 0.01:
        return "1°" if cells_per_degree == 1 else f"1/{cells_per_degree}°"
    return f"{cell_size_degrees:g}°"


def default_dataset_label(directory: Path, dataset_slug: str) -> str:
    """Catalog label "MODEL (resolution)", e.g. "GLONET (1/4°)", the resolution read from the maps manifest."""
    name = dataset_slug.removesuffix("_1_degree").replace("_", " ").upper()
    manifest = _read_json(directory / f"{dataset_slug}{PYRAMID_MANIFEST_SUFFIX}", {})
    cell_sizes = [level["cell_size_deg"] for level in manifest.get("levels", []) if level.get("cell_size_deg")]
    if cell_sizes:
        return f"{name} ({_resolution_label(min(cell_sizes))})"
    return f"{name} (1°)" if dataset_slug.endswith("_1_degree") else name


def update_viewer_indexes(
    directory: Path,
    dataset_slug: str,
    region: RegionSpec,
    artifacts: ViewerArtifacts,
    summary_rows: list[dict],
    kind: str = DEFAULT_CHALLENGER_KIND,
) -> None:
    """Merge the entries of one challenger and region into the index files, replacing its previous entries."""
    # Several challengers can be evaluated in parallel processes into the same directory.
    directory_descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(directory_descriptor, fcntl.LOCK_EX)
        if artifacts.has_maps:
            datasets_path = directory / DATASETS_FILENAME
            entries = {entry["slug"]: entry for entry in _read_json(datasets_path, {"datasets": []})["datasets"]}
            label = entries.get(dataset_slug, {}).get("label")
            entries[dataset_slug] = {
                "slug": dataset_slug,
                "label": label if label and label != dataset_slug else default_dataset_label(directory, dataset_slug),
                "kind": kind,
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
    matchups: pandas.DataFrame | Class4Matchups | None,
    document_rows: list[dict],
    start_tables: dict[str, list[pandas.DataFrame]] | None = None,
    map_dataset: xarray.Dataset | None = None,
) -> None:
    """
    Write the viewer data of one evaluated challenger and region into ``directory``.

    ``matchups`` are the Class IV match-ups the scores were computed from, ``None`` when the
    observations are unavailable, and ``document_rows`` the rows of the scores file.
    ``start_tables`` are the gridded score tables of each start date kept by the scoring run, by metric
    function name. Without them, the gridded metrics are computed again one start date at a time.
    """
    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    has_matchups = matchups is not None and len(matchups) > 0
    rows_by_start = class4_rmsd_rows_by_start(matchups) if has_matchups else None
    artifacts = write_viewer_artifacts(
        output_directory, challenger_name, challenger_dataset, region, matchups, rows_by_start, map_dataset
    )
    start_dates = [
        pandas.Timestamp(start_date) for start_date in challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    ]
    gridded_by_start = (
        gridded_scores_from_start_tables(start_tables)
        if start_tables
        else gridded_scores_by_start(challenger_dataset, region)
    )
    class4_by_start = class4_scores_by_start(rows_by_start, start_dates) if has_matchups else {}
    summary_rows = scores_summary_rows(
        challenger_name, region, document_rows, start_dates, gridded_by_start, class4_by_start
    )
    own_reference = get_challenger_own_reference(challenger_dataset)
    artifacts.insights["scores_by_start"] = write_scores_by_start(
        output_directory,
        scores_by_start_document(
            challenger_name,
            region.id,
            start_dates,
            gridded_by_start,
            class4_by_start if has_matchups else None,
            class4_bias_arrays(class4_bias_by_start(matchups), start_dates) if has_matchups else None,
            dropped_reference=None if own_reference is None else _SUMMARY_REFERENCES[own_reference],
        ),
    )
    if own_reference is not None:
        # A reference (GLORYS, GLO12 analysis) scored against its own gridded fields is zero by construction.
        summary_rows = [row for row in summary_rows if row["reference"] != _SUMMARY_REFERENCES[own_reference]]
    update_viewer_indexes(
        output_directory, challenger_name, region, artifacts, summary_rows, get_challenger_kind(challenger_dataset)
    )


def rebuild_eddy_census(
    directory: str, challenger_name: str, challenger_dataset: xarray.Dataset, region: RegionSpec
) -> None:
    """
    Rewrite only the eddy census of a challenger and region already in the viewer data ``directory``.

    Maps, water columns, match-ups and year files are left as they are; only the eddy files and the
    ``eddies`` entry of ``insights.json`` change. Useful after an eddy census format change, e.g.::

        from oceanbench.core.evaluate import open_challenger_dataset, resolve_evaluation_region
        dataset = open_challenger_dataset(open("glonet.py").read())
        rebuild_eddy_census("viewer", "glonet", dataset, resolve_evaluation_region("global"))
    """
    output_directory = Path(directory)
    insights_path = output_directory / INSIGHTS_FILENAME
    entries = _read_json(insights_path, {"datasets": {}})["datasets"]
    if region.id not in entries.get(challenger_name, {}):
        raise ValueError(f"No viewer data of {challenger_name} for region {region.id} in {directory}")
    insights_directory = output_directory / "insights" / challenger_name / region.id
    file_name = write_region_eddy_census(insights_directory, challenger_name, challenger_dataset, region)
    directory_descriptor = os.open(output_directory, os.O_RDONLY)
    try:
        fcntl.flock(directory_descriptor, fcntl.LOCK_EX)
        insights = _read_json(insights_path, {"datasets": {}})
        insights["datasets"][challenger_name][region.id]["eddies"] = (
            None if file_name is None else f"./data/insights/{challenger_name}/{region.id}/{file_name}"
        )
        _write_index(insights, insights_path)
    finally:
        os.close(directory_descriptor)
