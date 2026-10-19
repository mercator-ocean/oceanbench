# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import xarray

from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.marine_heatwaves import (
    _leap_year_day_of_year,
    detect_marine_heatwaves,
)

LEAD_DAYS = 17


def _temperature(values: numpy.ndarray, first_day: str = "2024-06-01") -> xarray.DataArray:
    pixels = values.reshape(-1, LEAD_DAYS)
    return xarray.DataArray(
        pixels.T.reshape(1, LEAD_DAYS, 1, pixels.shape[0]).astype(numpy.float32),
        dims=[
            Dimension.FIRST_DAY_DATETIME.key(),
            Dimension.LEAD_DAY_INDEX.key(),
            Dimension.LATITUDE.key(),
            Dimension.LONGITUDE.key(),
        ],
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): [numpy.datetime64(first_day)],
            Dimension.LEAD_DAY_INDEX.key(): numpy.arange(LEAD_DAYS),
            Dimension.LATITUDE.key(): [0.0],
            Dimension.LONGITUDE.key(): numpy.arange(pixels.shape[0], dtype=float),
        },
    )


def _climatology(
    longitudes: numpy.ndarray,
    seas_by_day: numpy.ndarray | float = 0.0,
    thresh_above_seas: float = 1.0,
) -> xarray.Dataset:
    seas = xarray.DataArray(
        numpy.broadcast_to(seas_by_day, (366,)).astype(float),
        dims=["doy"],
        coords={"doy": numpy.arange(1, 367)},
    ).expand_dims({Dimension.LATITUDE.key(): [0.0], Dimension.LONGITUDE.key(): longitudes})
    return xarray.Dataset({"seas": seas, "thresh": seas + thresh_above_seas})


def test_leap_year_day_of_year_puts_february_29_at_day_60_in_every_year():
    dates = xarray.DataArray(
        pandas.to_datetime(["2023-02-28", "2023-03-01", "2024-02-29", "2024-03-01", "2023-12-31"]).values,
        dims="date",
    )

    assert _leap_year_day_of_year(dates).values.tolist() == [59, 61, 60, 61, 366]


def test_detection_requires_five_days_and_gives_hobday_categories():
    values = numpy.zeros(LEAD_DAYS)
    values[2:8] = [1.5, 2.5, 3.5, 4.5, 1.2, 1.1]
    values[10:13] = 2.0
    temperature = _temperature(values)

    detection = detect_marine_heatwaves(temperature, _climatology(temperature.longitude.values)).squeeze()

    assert numpy.flatnonzero(detection.event.values).tolist() == [2, 3, 4, 5, 6, 7]
    assert detection.category.values[2:8].tolist() == [1, 2, 3, 4, 1, 1]
    assert int(detection.category.values[10:13].sum()) == 0
    numpy.testing.assert_allclose(detection.intensity.values, values, atol=1e-6)


def test_detection_joins_gaps_of_two_days():
    values = numpy.array([2.0] * 5 + [0.0] * 2 + [2.0] * 5 + [0.0] * 5)
    temperature = _temperature(values)

    detection = detect_marine_heatwaves(temperature, _climatology(temperature.longitude.values)).squeeze()

    assert numpy.flatnonzero(detection.event.values).tolist() == list(range(12))


def test_climatology_follows_valid_dates_across_february_29():
    temperature = _temperature(numpy.full(LEAD_DAYS, 100.0), first_day="2024-02-27")
    climatology = _climatology(temperature.longitude.values, seas_by_day=numpy.arange(1, 367))

    intensity = detect_marine_heatwaves(temperature, climatology).intensity.squeeze()

    expected_day_of_year = 58 + numpy.arange(LEAD_DAYS)
    numpy.testing.assert_allclose(intensity.values, 100.0 - expected_day_of_year, atol=1e-6)


def test_missing_pixels_have_no_event_and_no_intensity():
    values = numpy.vstack([numpy.full(LEAD_DAYS, 2.0), numpy.full(LEAD_DAYS, numpy.nan)])
    temperature = _temperature(values)

    detection = detect_marine_heatwaves(temperature, _climatology(temperature.longitude.values)).squeeze()

    assert bool(detection.event.isel(longitude=0).all())
    assert not bool(detection.event.isel(longitude=1).any())
    assert bool(detection.intensity.isel(longitude=1).isnull().all())
    assert int(detection.category.isel(longitude=1).sum()) == 0


def test_dask_chunked_input_gives_the_same_detection():
    random_values = numpy.random.default_rng(0).normal(1.0, 1.0, size=(6, LEAD_DAYS))
    temperature = _temperature(random_values)
    climatology = _climatology(temperature.longitude.values)

    eager = detect_marine_heatwaves(temperature, climatology)
    lazy = detect_marine_heatwaves(temperature.chunk({Dimension.LONGITUDE.key(): 2}), climatology)

    assert lazy.event.chunks is not None
    xarray.testing.assert_identical(eager, lazy.compute())
