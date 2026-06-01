# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
import inspect
from pathlib import Path

import nbformat
import numpy
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


def test_glonet_latest_default_opens_temporary_source_init_and_assigns_evaluation_init(
    tmp_path: Path, monkeypatch
) -> None:
    source_dataset = _forecast_dataset().isel({Dimension.FIRST_DAY_DATETIME.key(): 0}, drop=True)
    source_dataset = source_dataset.rename({Dimension.LEAD_DAY_INDEX.key(): "time"})
    forecast_path = tmp_path / "2026-06-01.zarr"
    source_dataset.to_zarr(forecast_path)
    monkeypatch.setattr(
        live_datasets,
        "_configured_live_glonet_forecast_zarr_template",
        lambda: f"file://{tmp_path}/{{date}}.zarr",
    )

    dataset = live_datasets.glonet_latest()

    assert dataset.sizes[Dimension.FIRST_DAY_DATETIME.key()] == 1
    assert dataset[Dimension.FIRST_DAY_DATETIME.key()].values[0] == numpy.datetime64("2026-05-13")
    assert dataset[Dimension.LEAD_DAY_INDEX.key()].values.tolist() == [0]


def test_live_evaluation_report_is_class4_only() -> None:
    report = evaluation_report.prepare_live_evaluation_report(
        _forecast_dataset(),
        observation_zarr_template="file:///tmp/synthetic_class4/{day}.zarr",
        observation_last_available_day="2026-05-23",
    )

    assert "glo12_zarr_template" not in inspect.signature(evaluation_report.prepare_live_evaluation_report).parameters
    assert not hasattr(report, "glo12_dataset")
    assert not hasattr(report, "glo12_variable_rmsd")
    assert not hasattr(report, "forecast_comparison_explorer")


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
    assert "Near-real-time forecast validation" in all_sources
    assert "Forecast validation setup" in all_sources
    assert "prepare_live_evaluation_report" in all_sources
    assert "evaluation_report.class4_observation.rmsd" in all_sources
    assert "evaluation_report.class4_observation_error_explorer" in all_sources
    assert "Live evaluation" not in all_sources
    assert "glo12" not in all_sources.lower()
    assert "glorys" not in all_sources.lower()
