# SPDX-FileCopyrightText: 2026 Mercator Ocean International
#
# SPDX-License-Identifier: EUPL-1.2
"""The local root store layout of the ensemble campaign helpers."""

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy
import pandas
import pytest
import xarray

from oceanbench.core.references import observations as observations_module

HELPER_ROOT = Path(__file__).resolve().parent.parent / "helper_scripts"


def _helper_module(relative_path: str, module_name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(module_name, HELPER_ROOT / relative_path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def class4_helper() -> ModuleType:
    return _helper_module("ensemble_class4/score_ensemble_class4.py", "score_ensemble_class4")


@pytest.fixture(scope="module")
def gridded_helper() -> ModuleType:
    return _helper_module("ensemble_gridded/score_ensemble_gridded_multi.py", "score_ensemble_gridded_multi")


def _tiny_forecast_store(
    directory: Path,
    start_date: pandas.Timestamp,
    member_count: int,
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
    with_member_dimension: bool = True,
) -> Path:
    days = pandas.date_range(start_date, periods=2, freq="D")
    latitude = numpy.array([-0.25, 0.25])
    longitude = numpy.array([0.25, 0.75])
    depth = numpy.array([0.494, 47.37])
    shape = (member_count, len(days), len(depth), len(latitude), len(longitude))
    surface_shape = (member_count, len(days), len(latitude), len(longitude))
    dataset = xarray.Dataset(
        {
            "thetao": (("member", "time", "depth", "latitude", "longitude"), numpy.full(shape, 12.0)),
            "so": (("member", "time", "depth", "latitude", "longitude"), numpy.full(shape, 35.0)),
            "zos": (("member", "time", "latitude", "longitude"), numpy.full(surface_shape, 0.1)),
            "uo": (("member", "time", "depth", "latitude", "longitude"), numpy.full(shape, 0.2)),
            "vo": (("member", "time", "depth", "latitude", "longitude"), numpy.full(shape, 0.3)),
        },
        coords={
            "member": numpy.arange(member_count),
            "time": days,
            "depth": depth,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    if not with_member_dimension:
        dataset = dataset.isel(member=0, drop=True)
    dataset = dataset.rename({"latitude": latitude_name, "longitude": longitude_name})
    store_path = directory / f"{start_date:%Y%m%d}.zarr"
    dataset.to_zarr(store_path)
    return store_path


def test_the_class4_helper_reads_a_start_out_of_a_local_directory(class4_helper, tmp_path):
    start_date = pandas.Timestamp("2024-01-03")
    _tiny_forecast_store(tmp_path, start_date, member_count=2)
    specification = dataclasses.replace(
        class4_helper.CHALLENGERS["glowens"],
        store_layout=class4_helper.STORE_LOCAL_ROOT,
        store_root=str(tmp_path),
        lead_days_count=2,
    )

    challenger, first_day = class4_helper._open_challenger_start(specification, start_date)

    assert first_day == start_date
    assert class4_helper.ENSEMBLE_DIMENSION in challenger.dims
    assert challenger.sizes[class4_helper.ENSEMBLE_DIMENSION] == 2
    assert sorted(challenger.data_vars) == ["so", "thetao", "uo", "vo", "zos"]
    assert float(challenger["thetao"].isel({class4_helper.ENSEMBLE_DIMENSION: 0}).max()) == 12.0


def test_the_gridded_helper_reads_a_start_out_of_a_local_directory(gridded_helper, tmp_path):
    start_date = pandas.Timestamp("2024-01-03")
    store_path = _tiny_forecast_store(tmp_path, start_date, member_count=2)
    specification = dataclasses.replace(
        gridded_helper.CHALLENGERS["glowens"], store_layout=gridded_helper.STORE_LOCAL_ROOT, store_root=str(tmp_path)
    )

    dataset, root = gridded_helper._open_challenger(specification, start_date, "thetao")

    assert root == str(store_path)
    assert dataset.sizes["member"] == 2
    assert float(dataset["zos"].max()) == 0.1


def test_the_glowens_specification_reads_the_served_ringc_stores(class4_helper, gridded_helper):
    for name in ("glowens", "glowens-fill"):
        specification = gridded_helper.CHALLENGERS[name]
        assert specification.name == "glowens"
        assert specification.version == "glowens_v5_ringC"
        assert specification.store_layout == gridded_helper.STORE_ML_FORECAST_DEV
        assert specification.store_root is None
        assert specification.member_count == 16
        assert specification.last_lead_day == 9
    assert class4_helper.CHALLENGERS["glowens"].name == "glowens"
    assert class4_helper.CHALLENGERS["glowens"].version == "glowens_v5_ringC"
    assert class4_helper.CHALLENGERS["glowens"].store_layout == class4_helper.STORE_ML_FORECAST_DEV
    assert class4_helper.CHALLENGERS["glowens"].store_root is None
    assert class4_helper.CHALLENGERS["glowens"].lead_days_count == 9

    root, group = gridded_helper._store_location(
        gridded_helper.CHALLENGERS["glowens"], pandas.Timestamp("2024-01-03"), "thetao", filesystem=None
    )
    assert root == "oceanbench-bucket/dev/ml-forecast-outputs/glowens_v5_ringC/20240103.zarr"
    assert group is None


def test_the_class4_helper_reads_a_store_that_names_its_axes_lat_and_lon(class4_helper, tmp_path):
    start_date = pandas.Timestamp("2024-01-03")
    _tiny_forecast_store(tmp_path, start_date, member_count=2, latitude_name="lat", longitude_name="lon")
    specification = dataclasses.replace(
        class4_helper.CHALLENGERS["glowens"],
        store_layout=class4_helper.STORE_LOCAL_ROOT,
        store_root=str(tmp_path),
        lead_days_count=2,
    )

    challenger, _ = class4_helper._open_challenger_start(specification, start_date)

    assert "latitude" in challenger.dims
    assert "longitude" in challenger.dims
    assert "lat" not in challenger.dims
    assert "lon" not in challenger.dims


def test_the_gridded_helper_reads_a_store_that_names_its_axes_lat_and_lon(gridded_helper, tmp_path):
    start_date = pandas.Timestamp("2024-01-03")
    _tiny_forecast_store(tmp_path, start_date, member_count=2, latitude_name="lat", longitude_name="lon")
    specification = dataclasses.replace(
        gridded_helper.CHALLENGERS["glowens"], store_layout=gridded_helper.STORE_LOCAL_ROOT, store_root=str(tmp_path)
    )

    dataset, _ = gridded_helper._open_challenger(specification, start_date, "thetao")

    assert "latitude" in dataset.dims
    assert "longitude" in dataset.dims
    assert "lat" not in dataset.dims
    assert "lon" not in dataset.dims


def test_the_gridded_helper_refuses_an_ensemble_store_that_carries_no_member_axis(gridded_helper, tmp_path):
    start_date = pandas.Timestamp("2024-01-03")
    _tiny_forecast_store(tmp_path, start_date, member_count=1, with_member_dimension=False)
    specification = dataclasses.replace(
        gridded_helper.CHALLENGERS["glowens"], store_layout=gridded_helper.STORE_LOCAL_ROOT, store_root=str(tmp_path)
    )

    with pytest.raises(ValueError, match="member"):
        gridded_helper._open_challenger(specification, start_date, "thetao")


def test_the_gridded_helper_gives_a_deterministic_store_a_member_axis_of_length_one(gridded_helper, tmp_path):
    start_date = pandas.Timestamp("2024-01-03")
    _tiny_forecast_store(tmp_path, start_date, member_count=1, with_member_dimension=False)
    specification = dataclasses.replace(
        gridded_helper.CHALLENGERS["glowens"],
        store_layout=gridded_helper.STORE_LOCAL_ROOT,
        store_root=str(tmp_path),
        member_count=1,
    )

    dataset, _ = gridded_helper._open_challenger(specification, start_date, "thetao")

    assert dataset.sizes["member"] == 1


def _tiny_gloens_week(lead_days_count: int, member_count: int = 2) -> xarray.Dataset:
    """A GloEns week as the library opener hands it back, with its time axis already an index."""
    latitude = numpy.array([-0.25, 0.25])
    longitude = numpy.array([0.25, 0.75])
    depth = numpy.array([0.494, 47.37])
    shape = (member_count, lead_days_count, len(depth), len(latitude), len(longitude))
    surface_shape = (member_count, lead_days_count, len(latitude), len(longitude))
    dimensions = ("member", "lead_day_index", "depth", "latitude", "longitude")
    return xarray.Dataset(
        {
            "thetao": (dimensions, numpy.full(shape, 12.0)),
            "zos": (tuple(name for name in dimensions if name != "depth"), numpy.full(surface_shape, 0.1)),
        },
        coords={
            "member": numpy.arange(member_count),
            "lead_day_index": numpy.arange(lead_days_count),
            "depth": depth,
            "latitude": latitude,
            "longitude": longitude,
        },
    )


def test_the_class4_helper_cuts_the_gloens_week_to_the_scored_horizon(class4_helper, monkeypatch):
    start_label = pandas.Timestamp("2024-01-04")
    monkeypatch.setattr(class4_helper, "_open_gloens_forecast_week", lambda _: _tiny_gloens_week(27))

    challenger, first_day = class4_helper._open_challenger_start(class4_helper.CHALLENGERS["gloens"], start_label)

    assert first_day == start_label + class4_helper.GLOENS_START_LABEL_TO_FIRST_DAY
    assert challenger.sizes["lead_day_index"] == class4_helper.CHALLENGERS["gloens"].lead_days_count == 10


def _coverage_ending_on(last_published_day: str):
    """An observation store whose published days stop on ``last_published_day``."""
    last_day = numpy.datetime64(last_published_day, "D")
    return lambda day_datetime: numpy.datetime64(day_datetime, "D") <= last_day


def test_the_class4_helper_scores_only_the_lead_days_that_have_observations(class4_helper, monkeypatch, capsys):
    start_label = pandas.Timestamp("2024-12-26")
    monkeypatch.setattr(class4_helper, "_open_gloens_forecast_week", lambda _: _tiny_gloens_week(10))
    monkeypatch.setattr(observations_module, "observation_day_is_published", _coverage_ending_on("2025-01-04"))

    challenger, first_day = class4_helper._open_challenger_start(class4_helper.CHALLENGERS["gloens"], start_label)
    cut = class4_helper._cut_to_observed_lead_days(challenger, first_day)

    assert first_day == pandas.Timestamp("2024-12-27")
    assert challenger.sizes["lead_day_index"] == 10
    assert cut.sizes["lead_day_index"] == 9
    assert "dropped lead days without observations: 2025-01-05" in capsys.readouterr().out


def test_the_class4_helper_keeps_the_whole_horizon_when_every_observation_day_exists(class4_helper, monkeypatch):
    start_label = pandas.Timestamp("2024-12-19")
    monkeypatch.setattr(class4_helper, "_open_gloens_forecast_week", lambda _: _tiny_gloens_week(10))
    monkeypatch.setattr(observations_module, "observation_day_is_published", _coverage_ending_on("2025-01-04"))

    challenger, first_day = class4_helper._open_challenger_start(class4_helper.CHALLENGERS["gloens"], start_label)
    cut = class4_helper._cut_to_observed_lead_days(challenger, first_day)

    assert cut is challenger
    assert cut.sizes["lead_day_index"] == 10
