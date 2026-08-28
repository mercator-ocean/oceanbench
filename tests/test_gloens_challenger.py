# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import replace
from datetime import datetime, timedelta

import numpy
import xarray

from oceanbench.core import challenger_datasets
from oceanbench.core.challenger_datasets import gloens
from oceanbench.core.classIV_support import CHALLENGER_INVERSE_BAROMETER_VARIABLES
from oceanbench.core.curvilinear_staging import (
    CURVILINEAR_CHALLENGERS,
    GLOENS_FORECAST_DAYS,
    GLOENS_LAND_SENTINELS,
    GLOENS_MEMBER_COUNT,
    GLOENS_SOURCE_NAME,
    GLOENS_SUBSURFACE_CONTENTS,
    GLOENS_SURFACE_CONTENT,
    gloens_companion_grid_store,
    gloens_store_url,
)
from oceanbench.core.dataset_source import get_dataset_source
from oceanbench.core.ensemble_gridded import ENSEMBLE_DIMENSION

FIRST_INITIALISATION = datetime(2024, 1, 4)
FIRST_DAY = FIRST_INITIALISATION + timedelta(days=1)
PUBLISHED_DAYS = 4
LEAD_DAYS = PUBLISHED_DAYS - 1
MEMBERS = 3
ROWS = 5
COLUMNS = 6
LEVELS = 2

TARGET_LATITUDE = numpy.arange(40.0, 40.51, 0.25)
TARGET_LONGITUDE = numpy.arange(10.0, 10.51, 0.25)

DEPTH_VALUES = numpy.array([0.5, 47.4], dtype="float32")

SUBSURFACE_VARIABLES = {
    "3DT-thetao": ("thetao", "deptht"),
    "3DT-so": ("so", "deptht"),
    "3DU-uo": ("uo", "depthu"),
    "3DV-vo": ("vo", "depthv"),
}


def _tracer_grid() -> tuple[numpy.ndarray, numpy.ndarray]:
    row = numpy.arange(ROWS, dtype="float64")
    column = numpy.arange(COLUMNS, dtype="float64")
    latitude = 39.9 + 0.15 * row[:, numpy.newaxis] + 0.01 * column[numpy.newaxis, :]
    longitude = 9.9 + 0.15 * column[numpy.newaxis, :] + 0.01 * row[:, numpy.newaxis]
    return latitude, longitude


def _surface_values(fill: float) -> numpy.ndarray:
    return numpy.full((PUBLISHED_DAYS, MEMBERS, ROWS, COLUMNS), fill, dtype="float64")


def _daily_ramp(fill: float) -> numpy.ndarray:
    """A field whose value names the published day it holds, so a day the opener drops shows up."""
    published_days = numpy.arange(PUBLISHED_DAYS, dtype="float64")
    return _surface_values(fill) + published_days[:, numpy.newaxis, numpy.newaxis, numpy.newaxis]


def _times(initialisation_datetime: datetime) -> numpy.ndarray:
    return numpy.array(
        [numpy.datetime64(initialisation_datetime + timedelta(days=day), "ns") for day in range(PUBLISHED_DAYS)]
    )


def _surface_store(initialisation_datetime: datetime) -> xarray.Dataset:
    latitude, longitude = _tracer_grid()
    surface_dimensions = ("time", "ens", "y", "x")
    return xarray.Dataset(
        {
            "tos": (surface_dimensions, _daily_ramp(12.0)),
            "zos": (surface_dimensions, _surface_values(0.3)),
            "ssh_ib": (surface_dimensions, _surface_values(0.02)),
        },
        coords={
            "time": _times(initialisation_datetime),
            "ens": numpy.arange(MEMBERS),
            "latitude": (("y", "x"), latitude.astype("float32")),
            "longitude": (("y", "x"), longitude.astype("float32")),
        },
    )


def _subsurface_store(initialisation_datetime: datetime, content: str) -> xarray.Dataset:
    variable_name, depth_name = SUBSURFACE_VARIABLES[content]
    missing = numpy.full((ROWS, COLUMNS), numpy.nan)
    values = numpy.full((PUBLISHED_DAYS, MEMBERS, LEVELS, ROWS, COLUMNS), 7.0, dtype="float64")
    return xarray.Dataset(
        {variable_name: (("time", "ens", depth_name, "y", "x"), values)},
        coords={
            "time": _times(initialisation_datetime),
            "ens": numpy.arange(MEMBERS),
            depth_name: DEPTH_VALUES,
            "latitude": (("y", "x"), missing),
            "longitude": (("y", "x"), missing),
            "nav_lat": (("y", "x"), missing),
            "nav_lon": (("y", "x"), missing),
            "time_centered": 0.0,
            "time_counter": 0.0,
        },
    )


def _published_stores(initialisation_datetime: datetime) -> dict[str, xarray.Dataset]:
    return {
        gloens_store_url(initialisation_datetime, GLOENS_SURFACE_CONTENT): _surface_store(initialisation_datetime),
        **{
            gloens_store_url(initialisation_datetime, content): _subsurface_store(initialisation_datetime, content)
            for content in GLOENS_SUBSURFACE_CONTENTS
        },
    }


def _open_published_store(store_url: str) -> xarray.Dataset:
    for initialisation_datetime in challenger_datasets._gloens_initialisation_datetimes():
        stores = _published_stores(initialisation_datetime)
        if store_url in stores:
            return stores[store_url]
    raise AssertionError(f"the opener asked for a store no initialisation publishes: {store_url}")


def _with_published_stores(monkeypatch) -> None:
    monkeypatch.setattr(challenger_datasets, "open_gloens_store", _open_published_store)


def _on_a_small_target_grid(monkeypatch) -> None:
    monkeypatch.setitem(
        CURVILINEAR_CHALLENGERS,
        GLOENS_SOURCE_NAME,
        replace(
            CURVILINEAR_CHALLENGERS[GLOENS_SOURCE_NAME],
            target_latitude=TARGET_LATITUDE,
            target_longitude=TARGET_LONGITUDE,
        ),
    )


def _week(first_day_datetime: datetime = FIRST_DAY) -> xarray.Dataset:
    return challenger_datasets._open_gloens_forecast_week(first_day_datetime)


def test_a_gloens_store_is_named_after_the_initialisation_and_the_content_it_holds():
    assert gloens_store_url(FIRST_INITIALISATION, "3DT-thetao") == (
        "https://s3.waw3-1.cloudferro.com/MOISICEEF/"
        "glo4-ens50_ng_1d-m_20240104-20240131_3DT-thetao_fcst_R20240104.zarr"
    )


def test_the_companion_grid_store_is_the_surface_content_of_the_same_initialisation():
    assert gloens_companion_grid_store(FIRST_INITIALISATION) == gloens_store_url(
        FIRST_INITIALISATION, GLOENS_SURFACE_CONTENT
    )


def test_the_gloens_initialisations_are_the_thursdays_of_the_year():
    initialisation_datetimes = challenger_datasets._gloens_initialisation_datetimes()

    assert len(initialisation_datetimes) == 52
    assert initialisation_datetimes[0] == FIRST_INITIALISATION
    assert initialisation_datetimes[-1] == datetime(2024, 12, 26)
    assert {initialisation_datetime.weekday() for initialisation_datetime in initialisation_datetimes} == {3}


def test_a_gloens_week_is_the_five_published_stores_read_as_one_dataset(monkeypatch):
    _with_published_stores(monkeypatch)

    week = _week()

    assert set(week.data_vars) == {"tos", "zos", "ssh_ib", "thetao", "so", "uo", "vo"}


def test_a_gloens_week_reads_its_time_axis_as_the_lead_day_index(monkeypatch):
    _with_published_stores(monkeypatch)

    week = _week()

    assert list(week["lead_day_index"].values) == list(range(LEAD_DAYS))
    assert "time" not in week.dims


def test_gloens_lead_day_one_is_the_forecast_day_after_the_initialisation_not_its_nowcast(monkeypatch):
    """The convention fingerprint, which no other assertion of this file can fail on.

    GloEns is the only challenger of the benchmark that publishes a daily mean for the day it
    starts from. Reading that nowcast as lead day one would score this challenger a day ahead
    of every other one, and the library reads consistently either way, so the reading is pinned
    here against the two facts that distinguish it: the week starts the day after the
    initialisation, and the field the store holds for the initialisation day is not in it.
    """
    _with_published_stores(monkeypatch)

    first_day_datetimes = challenger_datasets._gloens_first_day_datetimes()
    week = _week()

    assert first_day_datetimes[0] == FIRST_INITIALISATION + timedelta(days=1)
    assert {first_day_datetime.weekday() for first_day_datetime in first_day_datetimes} == {4}
    assert week.sizes["lead_day_index"] == PUBLISHED_DAYS - 1
    assert week["tos"].values.min() == numpy.float32(13.0)


def test_the_gloens_members_carry_the_name_the_ensemble_metrics_read(monkeypatch):
    _with_published_stores(monkeypatch)

    week = _week()

    assert week.sizes[ENSEMBLE_DIMENSION] == MEMBERS
    assert "ens" not in week.dims


def test_the_staggered_gloens_depths_collapse_onto_the_tracer_axis(monkeypatch):
    _with_published_stores(monkeypatch)

    week = _week()

    assert [name for name in week.dims if "depth" in str(name)] == ["depth"]
    numpy.testing.assert_array_equal(week["depth"].values, DEPTH_VALUES)
    for variable_name in ("thetao", "so", "uo", "vo"):
        assert "depth" in week[variable_name].dims


def test_a_gloens_week_is_read_as_float32(monkeypatch):
    _with_published_stores(monkeypatch)

    week = _week()

    assert {str(week[name].dtype) for name in week.data_vars} == {"float32"}


def test_the_gloens_week_keeps_the_grid_of_the_store_that_describes_it(monkeypatch):
    _with_published_stores(monkeypatch)

    week = _week()

    assert numpy.isfinite(week["latitude"].values).all()
    assert "nav_lat" not in week.variables
    assert "time_counter" not in week.variables


def test_the_gloens_week_leaves_its_inverse_barometer_beside_its_sea_level(monkeypatch):
    _with_published_stores(monkeypatch)

    week = _week()

    assert CHALLENGER_INVERSE_BAROMETER_VARIABLES[GLOENS_SOURCE_NAME] in week.data_vars
    assert week["zos"].values.max() == numpy.float32(0.3)


def test_the_gloens_challenger_is_read_through_onto_the_scoring_grid(monkeypatch):
    _with_published_stores(monkeypatch)
    _on_a_small_target_grid(monkeypatch)
    monkeypatch.setattr(
        challenger_datasets,
        "_gloens_first_day_datetimes",
        lambda: [FIRST_DAY, FIRST_DAY + timedelta(days=7)],
    )

    challenger_dataset = gloens()

    assert get_dataset_source(challenger_dataset).name == GLOENS_SOURCE_NAME
    assert challenger_dataset.sizes["first_day_datetime"] == 2
    assert challenger_dataset.sizes[ENSEMBLE_DIMENSION] == MEMBERS
    numpy.testing.assert_array_equal(challenger_dataset["latitude"].values, TARGET_LATITUDE)
    numpy.testing.assert_array_equal(challenger_dataset["longitude"].values, TARGET_LONGITUDE)
    assert "y" not in challenger_dataset.dims


def test_the_published_gloens_facts_the_opener_is_built_on():
    assert GLOENS_FORECAST_DAYS == 28
    assert GLOENS_MEMBER_COUNT == 50
    assert GLOENS_LAND_SENTINELS["tos"] == 17.5
