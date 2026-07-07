# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Reconciliation harness math against tiny synthetic parquet/JSON fixtures.

No real challenger data is downloaded; a small match-up frame is written to the served parquet
layout and hand-derived aggregates (pooled Class-4 RMSD, per-start year RMSD/bias, per-cell error
geography) are checked to reconcile, and a deliberately corrupted aggregate to fail.
"""

import json
from pathlib import Path

import numpy
import pandas
import pytest

from oceanbench.publish import viewer_artifacts
from oceanbench.publish.reconcile import ReconciliationError, reconcile_viewer_artifacts

_TARGETS = [
    ("sea_surface_height_above_geoid", "surface", "SSH"),
    ("eastward_sea_water_velocity", "15m", "u"),
]


def _matchup_frame() -> pandas.DataFrame:
    generator = numpy.random.default_rng(11)
    rows = []
    for start_date in ("2024-01-03", "2024-01-10"):
        for lead_day in (1, 5):
            for variable, depth_bin, _ in _TARGETS:
                for _ in range(30):
                    rows.append(
                        {
                            "variable": variable,
                            "depth_bin": depth_bin,
                            "lead_day": lead_day,
                            "start_date": numpy.datetime64(start_date),
                            "latitude": float(generator.uniform(-70, 70)),
                            "longitude": float(generator.uniform(-180, 180)),
                            "observation_value": float(generator.normal()),
                            "model_value": float(generator.normal()),
                        }
                    )
    return pandas.DataFrame(rows)


def _pooled_rmsd(group: pandas.DataFrame) -> float:
    error = group["model_value"] - group["observation_value"]
    return float(numpy.sqrt((error * error).mean()))


def _class4_summary(frame: pandas.DataFrame, dataset: str, region: str) -> list[dict]:
    records = []
    for (variable, depth_bin, lead_day), group in frame.groupby(["variable", "depth_bin", "lead_day"]):
        records.append(
            {
                "metric": "class4_rmsd",
                "challenger": dataset,
                "region": region,
                "variable": variable,
                "depth": depth_bin,
                "lead_day": int(lead_day),
                "mean": _pooled_rmsd(group),
            }
        )
    return records


def _year_rmsd_json(frame: pandas.DataFrame) -> dict:
    variables = {}
    for variable, depth_bin, short in _TARGETS:
        leads = {}
        for lead_day in (1, 5):
            subset = frame[(frame["variable"] == variable) & (frame["lead_day"] == lead_day)]
            dates, rmsd, counts, bias = [], [], [], []
            start_labels = subset["start_date"].astype("datetime64[ns]").dt.strftime("%Y-%m-%d")
            for start_date, group in subset.groupby(start_labels):
                error = group["model_value"] - group["observation_value"]
                dates.append(start_date)
                rmsd.append(round(float(numpy.sqrt((error * error).mean())), 6))
                counts.append(int(len(group)))
                bias.append(round(float(error.mean()), 6))
            leads[str(lead_day)] = {"dates": dates, "rmsd": rmsd, "n": counts, "bias": bias}
        variables[short] = {"depth_bin": depth_bin, "leads": leads}
    return {"variables": variables, "meta": {"method": "pooled"}}


def _year_geography_json(frame: pandas.DataFrame, grid: dict) -> dict:
    cell_count = grid["nlat"] * grid["nlon"]
    variables = {}
    for variable, depth_bin, short in _TARGETS:
        decimals = viewer_artifacts._YEAR_GEOGRAPHY_DECIMALS[short]
        leads = {}
        for lead_day in (1, 5):
            subset = frame[(frame["variable"] == variable) & (frame["lead_day"] == lead_day)]
            absolute_sum = numpy.zeros(cell_count)
            counts = numpy.zeros(cell_count, dtype=numpy.int64)
            cell, valid = viewer_artifacts._grid_cells(
                subset["latitude"].to_numpy(), subset["longitude"].to_numpy(), grid
            )
            error = numpy.abs((subset["model_value"] - subset["observation_value"]).to_numpy())
            numpy.add.at(absolute_sum, cell[valid], error[valid])
            numpy.add.at(counts, cell[valid], 1)
            leads[str(lead_day)] = [
                None if counts[index] == 0 else round(float(absolute_sum[index] / counts[index]), decimals)
                for index in range(cell_count)
            ]
        variables[short] = {"leads": leads}
    return {"grid": grid, "variables": variables, "meta": {"aggregation": "time-mean of |obs-model| per cell"}}


def _write_tree(tmp_path: Path, dataset: str, region: str, summary: list[dict]) -> tuple[str, dict]:
    data_directory = tmp_path / "data"
    insights_directory = data_directory / "insights" / dataset / region
    insights_directory.mkdir(parents=True, exist_ok=True)
    frame = _matchup_frame()
    viewer_artifacts.write_matchup_parquet(frame, str(insights_directory / "class4-matchups.parquet"))

    grid = viewer_artifacts._year_grid_for_region(region)
    (insights_directory / "year-rmsd-by-start.json").write_text(json.dumps(_year_rmsd_json(frame)), encoding="utf-8")
    (insights_directory / "year-error-geography.json").write_text(
        json.dumps(_year_geography_json(frame, grid)), encoding="utf-8"
    )
    prefix = f"./data/insights/{dataset}/{region}"
    insights = {
        "datasets": {
            dataset: {
                region: {
                    "class4_matchups": f"{prefix}/class4-matchups.parquet",
                    "year_rmsd_by_start": f"{prefix}/year-rmsd-by-start.json",
                    "year_error_geography": f"{prefix}/year-error-geography.json",
                    "eddies": None,
                }
            }
        },
        "scores_summary": "./data/scores-summary.json",
    }
    (data_directory / "insights.json").write_text(json.dumps(insights), encoding="utf-8")
    (data_directory / "scores-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return str(data_directory), {"frame": frame, "grid": grid}


def test_reconcile_passes_on_consistent_artifacts(tmp_path) -> None:
    dataset, region = "synthetic", "global"
    summary = _class4_summary(_matchup_frame(), dataset, region)
    base, _ = _write_tree(tmp_path, dataset, region, summary)

    report = reconcile_viewer_artifacts(base, dataset=dataset, region=region)

    assert report["passed"] is True
    assert report["checks_total"] > 0
    assert report["checks_passed"] == report["checks_total"]
    check_kinds = {check["check"] for entry in report["datasets"] for check in entry["checks"]}
    assert {"class4_pooled_rmsd", "year_rmsd_by_start", "year_error_geography"} <= check_kinds
    assert Path(report["report_path"]).exists()


def test_reconcile_flags_corrupted_class4_aggregate(tmp_path) -> None:
    dataset, region = "synthetic", "global"
    summary = _class4_summary(_matchup_frame(), dataset, region)
    summary[0]["mean"] *= 1.05
    base, _ = _write_tree(tmp_path, dataset, region, summary)

    with pytest.raises(ReconciliationError):
        reconcile_viewer_artifacts(base, dataset=dataset, region=region, output_path=str(tmp_path / "report.json"))

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    failed = [check for entry in report["datasets"] for check in entry["checks"] if not check["passed"]]
    assert any(check["check"] == "class4_pooled_rmsd" for check in failed)
