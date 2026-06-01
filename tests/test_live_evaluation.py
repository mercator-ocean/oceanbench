# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
from pathlib import Path

import nbformat
import numpy
import pandas
import xarray

from oceanbench.core import evaluation_report, live_datasets
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.python2jupyter import generate_live_evaluation_notebook_file


def _forecast_dataset(first_day_count: int = 1) -> xarray.Dataset:
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    lead_day_key = Dimension.LEAD_DAY_INDEX.key()
    depth_key = Dimension.DEPTH.key()
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    dimensions_3d = (first_day_key, lead_day_key, depth_key, latitude_key, longitude_key)
    dimensions_2d = (first_day_key, lead_day_key, latitude_key, longitude_key)
    coords = {
        first_day_key: numpy.array(["2026-05-13"], dtype="datetime64[ns]")[:first_day_count],
        lead_day_key: [0],
        depth_key: [0.5],
        latitude_key: [0.0, 1.0],
        longitude_key: [10.0, 11.0],
    }
    three_dimensional_values = numpy.zeros((first_day_count, 1, 1, 2, 2))
    two_dimensional_values = numpy.zeros((first_day_count, 1, 2, 2))
    return xarray.Dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (dimensions_2d, two_dimensional_values),
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): (dimensions_3d, three_dimensional_values),
            Variable.SEA_WATER_SALINITY.key(): (dimensions_3d, three_dimensional_values),
            Variable.EASTWARD_SEA_WATER_VELOCITY.key(): (dimensions_3d, three_dimensional_values),
            Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): (dimensions_3d, three_dimensional_values),
        },
        coords=coords,
    )


def test_glonet_latest_loader_opens_one_local_forecast_init(tmp_path: Path) -> None:
    source_dataset = _forecast_dataset().isel({Dimension.FIRST_DAY_DATETIME.key(): 0}, drop=True)
    source_dataset = source_dataset.rename({Dimension.LEAD_DAY_INDEX.key(): "time"})
    forecast_path = tmp_path / "2026-05-13.zarr"
    source_dataset.to_zarr(forecast_path)

    dataset = live_datasets.glonet_latest(
        first_day_datetime=datetime(2026, 5, 13),
        zarr_template=f"file://{tmp_path}/{{date}}.zarr",
    )

    assert dataset.sizes[Dimension.FIRST_DAY_DATETIME.key()] == 1
    assert dataset[Dimension.FIRST_DAY_DATETIME.key()].values[0] == numpy.datetime64("2026-05-13")
    assert dataset[Dimension.LEAD_DAY_INDEX.key()].values.tolist() == [0]


def test_live_first_day_datetime_can_be_pinned(monkeypatch) -> None:
    monkeypatch.setenv("OCEANBENCH_LIVE_FIRST_DAY", "2026-05-20")

    assert live_datasets.live_first_day_datetime() == datetime(2026, 5, 20)


def test_live_first_day_datetime_defaults_to_latest_fully_evaluable_init(monkeypatch) -> None:
    monkeypatch.delenv("OCEANBENCH_LIVE_FIRST_DAY", raising=False)
    monkeypatch.setenv("OCEANBENCH_LIVE_OBSERVATION_LAST_DAY", "2026-05-23")

    assert live_datasets.live_first_day_datetime() == datetime(2026, 5, 13)


def test_live_evaluation_report_uses_glo12_without_glorys(monkeypatch) -> None:
    calls = {"glorys": 0, "glo12": 0}

    def glorys_reanalysis_dataset(_):
        calls["glorys"] += 1
        raise AssertionError("live evaluation must not use GLORYS")

    def glo12_analysis_dataset(_):
        calls["glo12"] += 1
        return _forecast_dataset()

    monkeypatch.setattr(evaluation_report, "glorys_reanalysis_dataset", glorys_reanalysis_dataset)
    monkeypatch.setattr(evaluation_report, "glo12_analysis_dataset", glo12_analysis_dataset)
    monkeypatch.setattr(evaluation_report, "compute_mixed_layer_depth", lambda dataset: dataset)
    monkeypatch.setattr(evaluation_report, "compute_geostrophic_currents", lambda dataset: dataset)
    monkeypatch.setattr(evaluation_report, "rmsd", lambda **_: pandas.DataFrame({"Lead day 1": [0.0]}))

    report = evaluation_report.prepare_live_evaluation_report(
        _forecast_dataset(),
        observation_zarr_template="file:///tmp/synthetic_class4/{day}.zarr",
        observation_last_available_day="2026-05-23",
    )

    assert report.reference_datasets == {evaluation_report.GLO12_REFERENCE_NAME: report.glo12_dataset}
    assert report.glo12_variable_rmsd.to_dict(orient="list") == {"Lead day 1": [0.0]}
    assert calls == {"glorys": 0, "glo12": 1}


def test_live_evaluation_report_can_use_configured_glo12_reference(tmp_path: Path, monkeypatch) -> None:
    source_dataset = _forecast_dataset().isel({Dimension.FIRST_DAY_DATETIME.key(): 0}, drop=True)
    source_dataset = source_dataset.rename({Dimension.LEAD_DAY_INDEX.key(): "time"})
    reference_path = tmp_path / "2026-05-13.zarr"
    source_dataset.to_zarr(reference_path)

    def glo12_analysis_dataset(_):
        raise AssertionError("configured live GLO12 reference should be used")

    monkeypatch.setattr(evaluation_report, "glo12_analysis_dataset", glo12_analysis_dataset)
    monkeypatch.setattr(evaluation_report, "compute_mixed_layer_depth", lambda dataset: dataset)
    monkeypatch.setattr(evaluation_report, "compute_geostrophic_currents", lambda dataset: dataset)
    monkeypatch.setattr(evaluation_report, "rmsd", lambda **_: pandas.DataFrame({"Lead day 1": [0.0]}))

    report = evaluation_report.prepare_live_evaluation_report(
        _forecast_dataset(),
        glo12_zarr_template=f"file://{tmp_path}/{{date}}.zarr",
    )

    assert report.glo12_dataset.sizes[Dimension.FIRST_DAY_DATETIME.key()] == 1
    assert report.glo12_variable_rmsd.to_dict(orient="list") == {"Lead day 1": [0.0]}


def test_generate_live_evaluation_notebook_excludes_glorys(tmp_path: Path) -> None:
    challenger_path = tmp_path / "glonet_latest.py"
    challenger_path.write_text(
        "import oceanbench\n\nchallenger_dataset = oceanbench.datasets.challenger.glonet_latest()\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "glonet.latest.global.report.ipynb"

    generate_live_evaluation_notebook_file(
        str(challenger_path),
        str(output_path),
        region="global",
    )

    notebook = nbformat.read(output_path, as_version=4)
    all_sources = "\n".join(cell.source for cell in notebook.cells)

    assert notebook.metadata["oceanbench"]["live_evaluation"] is True
    assert "Live evaluation" in all_sources
    assert "prepare_live_evaluation_report" in all_sources
    assert "live_glo12_analysis_zarr_template" in all_sources
    assert "evaluation_report.class4_observation.rmsd" in all_sources
    assert "evaluation_report.class4_observation_error_explorer" in all_sources
    assert "evaluation_report.glo12_variable_rmsd" in all_sources
    assert "evaluation_report.glo12_mixed_layer_depth_rmsd" in all_sources
    assert "evaluation_report.glo12_geostrophic_current_rmsd" in all_sources
    assert "evaluation_report.forecast_comparison_explorer" in all_sources
    assert "evaluation_report.dynamic_diagnostic_explorer" in all_sources
    assert "evaluation_report.zonal_psd_explorer" in all_sources
    assert "glorys" not in all_sources.lower()
