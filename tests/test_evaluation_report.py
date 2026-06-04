# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import xarray
import json

from oceanbench.core import evaluation_report
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.references.observations import ObservationDataUnavailableError


def _dataset() -> xarray.Dataset:
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    lead_day_key = Dimension.LEAD_DAY_INDEX.key()
    depth_key = Dimension.DEPTH.key()
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    return xarray.Dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (
                (first_day_key, lead_day_key, latitude_key, longitude_key),
                numpy.zeros((1, 1, 2, 2)),
            ),
        },
        coords={
            first_day_key: numpy.array(["2024-01-01"], dtype="datetime64[ns]"),
            lead_day_key: [0],
            depth_key: [0.0],
            latitude_key: [0.0, 1.0],
            longitude_key: [0.0, 1.0],
        },
    )


def _mld_dataset(source: xarray.Dataset) -> xarray.Dataset:
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    lead_day_key = Dimension.LEAD_DAY_INDEX.key()
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    return xarray.Dataset(
        {
            Variable.MIXED_LAYER_DEPTH.key(): (
                (first_day_key, lead_day_key, latitude_key, longitude_key),
                numpy.ones((1, 1, 2, 2)),
            ),
        },
        coords={key: source[key] for key in [first_day_key, lead_day_key, latitude_key, longitude_key]},
    )


def _geostrophic_dataset(source: xarray.Dataset) -> xarray.Dataset:
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    lead_day_key = Dimension.LEAD_DAY_INDEX.key()
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    dimensions = (first_day_key, lead_day_key, latitude_key, longitude_key)
    coords = {key: source[key] for key in [first_day_key, lead_day_key, latitude_key, longitude_key]}
    return xarray.Dataset(
        {
            Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY.key(): (dimensions, numpy.ones((1, 1, 2, 2))),
            Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY.key(): (dimensions, numpy.ones((1, 1, 2, 2))),
        },
        coords=coords,
    )


def test_evaluation_report_reuses_reference_and_derived_datasets(monkeypatch) -> None:
    challenger_dataset = _dataset()
    glorys_dataset = _dataset()
    glo12_dataset = _dataset()
    calls = {
        "glorys": 0,
        "glo12": 0,
        "mld": 0,
        "geostrophic": 0,
    }

    def glorys_reanalysis_dataset(_):
        calls["glorys"] += 1
        return glorys_dataset

    def glo12_analysis_dataset(_):
        calls["glo12"] += 1
        return glo12_dataset

    def compute_mixed_layer_depth(dataset):
        calls["mld"] += 1
        return _mld_dataset(dataset)

    def compute_geostrophic_currents(dataset):
        calls["geostrophic"] += 1
        return _geostrophic_dataset(dataset)

    monkeypatch.setattr(evaluation_report, "glorys_reanalysis_dataset", glorys_reanalysis_dataset)
    monkeypatch.setattr(evaluation_report, "glo12_analysis_dataset", glo12_analysis_dataset)
    monkeypatch.setattr(evaluation_report, "compute_mixed_layer_depth", compute_mixed_layer_depth)
    monkeypatch.setattr(evaluation_report, "compute_geostrophic_currents", compute_geostrophic_currents)
    monkeypatch.setattr(
        evaluation_report,
        "rmsd",
        lambda **_: pandas.DataFrame({"Lead day 1": [0.0]}),
    )

    report = evaluation_report.prepare_evaluation_report(challenger_dataset)

    assert report.glorys_dataset is report.glorys_dataset
    assert report.glo12_dataset is report.glo12_dataset
    report.glorys_mixed_layer_depth_rmsd
    report.glorys_geostrophic_current_rmsd
    report.glorys_dynamic_dataset
    report.glo12_dynamic_dataset

    assert calls == {
        "glorys": 1,
        "glo12": 1,
        "mld": 3,
        "geostrophic": 3,
    }
    assert set(report.reference_datasets) == {
        evaluation_report.GLORYS_REFERENCE_NAME,
        evaluation_report.GLO12_REFERENCE_NAME,
    }


def test_evaluation_report_handles_unavailable_class4_observations(monkeypatch) -> None:
    calls = {"observations": 0}

    def unavailable_observations(_):
        calls["observations"] += 1
        raise ObservationDataUnavailableError("Class IV unavailable")

    monkeypatch.setattr(evaluation_report, "observations", unavailable_observations)

    report = evaluation_report.prepare_evaluation_report(_dataset())

    assert report.class4_observation.rmsd.to_dict(orient="list") == {"Message": ["Class IV unavailable"]}
    assert report.class4_observation_error_explorer is None
    assert report.class4_observation.rmsd.to_dict(orient="list") == {"Message": ["Class IV unavailable"]}
    assert calls == {"observations": 1}


def test_evaluation_report_writes_scores_json_from_metric_dataframes(monkeypatch, tmp_path) -> None:
    def score_dataframes(_):
        return {
            "rmsd_variables_glorys": pandas.DataFrame(
                {"Lead day 1": [1.1], "Lead day 2": [1.2]},
                index=["Temperature (C) [sea_water_potential_temperature]{100m}"],
            ),
            "lagrangian_glorys": pandas.DataFrame(
                {"Lead day 2": [2.2]},
                index=["Lagrangian trajectory deviation (km) []"],
            ),
            "rmsd_variables_observations": pandas.DataFrame({"Message": ["Class IV unavailable"]}),
        }

    monkeypatch.setattr(evaluation_report.EvaluationReportContext, "score_dataframes", score_dataframes)
    report = evaluation_report.prepare_evaluation_report(_dataset())
    scores_path = tmp_path / "glonet.global.scores.json"

    report.write_scores_json(scores_path, "glonet")

    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    assert scores["rmsd_variables_glorys"]["depths"]["100m"]["variables"]["temperature"] == {
        "standard_name": "sea_water_potential_temperature",
        "unit": "C",
        "data": {"1": 1.1, "2": 1.2},
    }
    assert scores["lagrangian_glorys"]["depths"]["flat"]["variables"]["lagrangian trajectory deviation"]["data"] == {
        "2": 2.2
    }
    assert "rmsd_variables_observations" not in scores


def test_evaluation_report_lagrangian_widget_uses_display_particle_count(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(evaluation_report, "glorys_reanalysis_dataset", lambda _: _dataset())
    monkeypatch.setattr(evaluation_report, "glo12_analysis_dataset", lambda _: _dataset())

    def plot_multi_reference_lagrangian_trajectory_explorer(_, reference_datasets, particle_count):
        captured["references"] = tuple(reference_datasets)
        captured["particle_count"] = particle_count
        return "lagrangian-widget"

    monkeypatch.setattr(
        evaluation_report.visualization,
        "plot_multi_reference_lagrangian_trajectory_explorer",
        plot_multi_reference_lagrangian_trajectory_explorer,
    )

    report = evaluation_report.prepare_evaluation_report(_dataset())

    assert report.lagrangian_trajectory_explorer == "lagrangian-widget"
    assert captured == {
        "references": (
            evaluation_report.GLORYS_REFERENCE_NAME,
            evaluation_report.GLO12_REFERENCE_NAME,
        ),
        "particle_count": 1000,
    }
