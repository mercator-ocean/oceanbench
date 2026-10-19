# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import heatwave3
import numpy
import xarray

from oceanbench.core.dataset_utils import Dimension

CLIMATOLOGY_DAY_OF_YEAR_DIMENSION = "doy"
CLIMATOLOGY_MEAN_VARIABLE = "seas"
CLIMATOLOGY_THRESHOLD_VARIABLE = "thresh"

_DAYS_BEFORE_MONTH_IN_LEAP_YEAR = numpy.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])


def detect_marine_heatwaves(
    temperature: xarray.DataArray,
    climatology: xarray.Dataset,
) -> xarray.Dataset:
    """
    Detect marine heatwave days in forecast windows with heatwave3.

    Each forecast window (one ``first_day_datetime``) is detected on its own along
    ``lead_day_index``, following Hobday et al. (2016) with heatwave3 defaults: at least five
    consecutive days above the threshold, gaps of at most two days joined.

    Parameters
    ----------
    temperature : xarray.DataArray
        Daily temperature with ``first_day_datetime`` and consecutive ``lead_day_index`` days.
    climatology : xarray.Dataset
        heatwave3 climatology with ``seas`` and ``thresh`` on a 366-day ``doy`` dimension
        (leap-year day of year) and the spatial grid of ``temperature``.

    Returns
    -------
    xarray.Dataset
        ``event`` (bool), ``intensity`` (temperature minus ``seas``, every day with data) and
        ``category`` (Hobday et al. 2018 category 1 to 4 on event days, 0 otherwise), with the
        dimensions of ``temperature``.
    """
    lead_day_dimension = Dimension.LEAD_DAY_INDEX.key()
    climatology_on_valid_dates = _climatology_on_valid_dates(climatology, temperature)
    event, intensity, category = xarray.apply_ufunc(
        heatwave3.detect_event_array3,
        temperature,
        climatology_on_valid_dates[CLIMATOLOGY_MEAN_VARIABLE],
        climatology_on_valid_dates[CLIMATOLOGY_THRESHOLD_VARIABLE],
        input_core_dims=[[lead_day_dimension]] * 3,
        output_core_dims=[[lead_day_dimension]] * 3,
        kwargs={"outputs": ("event", "intensity", "category"), "n_threads": 1},
        dask="parallelized",
        output_dtypes=[bool, numpy.float32, numpy.int8],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    return xarray.Dataset({"event": event, "intensity": intensity, "category": category}).transpose(*temperature.dims)


def _climatology_on_valid_dates(climatology: xarray.Dataset, temperature: xarray.DataArray) -> xarray.Dataset:
    day_of_year = _leap_year_day_of_year(_valid_dates(temperature))
    return (
        climatology[[CLIMATOLOGY_MEAN_VARIABLE, CLIMATOLOGY_THRESHOLD_VARIABLE]]
        .sel({CLIMATOLOGY_DAY_OF_YEAR_DIMENSION: day_of_year})
        .drop_vars(CLIMATOLOGY_DAY_OF_YEAR_DIMENSION)
    )


def _valid_dates(temperature: xarray.DataArray) -> xarray.DataArray:
    first_days = temperature[Dimension.FIRST_DAY_DATETIME.key()]
    lead_days = temperature[Dimension.LEAD_DAY_INDEX.key()]
    return first_days + lead_days.astype("timedelta64[D]")


def _leap_year_day_of_year(dates: xarray.DataArray) -> xarray.DataArray:
    return xarray.DataArray(
        _DAYS_BEFORE_MONTH_IN_LEAP_YEAR[dates.dt.month.values - 1] + dates.dt.day.values,
        dims=dates.dims,
        coords=dates.coords,
    )
