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
MINIMUM_CATEGORIES = [2, 3, 4]
EVENT_AGREEMENT_GROUPS = ["both", "challenger_only", "reference_only", "neither"]

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


def marine_heatwave_scores(
    challenger_temperature: xarray.DataArray,
    reference_temperature: xarray.DataArray,
    challenger_detection: xarray.Dataset,
    reference_detection: xarray.Dataset,
) -> xarray.Dataset:
    """
    Score challenger marine heatwave days against reference marine heatwave days.

    Every score is reduced over all dimensions except ``lead_day_index`` with
    cos(latitude) area weights, on the days where both temperatures exist. A zero
    denominator gives a missing score.

    Parameters
    ----------
    challenger_temperature, reference_temperature : xarray.DataArray
        Daily temperature on the same grid, forecast starts and lead days.
    challenger_detection, reference_detection : xarray.Dataset
        Outputs of :func:`detect_marine_heatwaves` for each temperature.

    Returns
    -------
    xarray.Dataset
        ``probability_of_detection``, ``false_alarm_ratio`` and ``critical_success_index`` of
        event days; the same three for days of at least each category in ``minimum_category``
        (2, 3, 4); ``category_accuracy`` over days where either side has an event;
        ``temperature_difference_rmse`` per ``event_agreement`` group (both, challenger only,
        reference only, neither), where the intensity difference reduces to the temperature
        difference; and ``temperature_bias`` (challenger minus reference).
    """
    valid = challenger_temperature.notnull() & reference_temperature.notnull()
    challenger_event = challenger_detection.event & valid
    reference_event = reference_detection.event & valid
    minimum_category = xarray.DataArray(
        MINIMUM_CATEGORIES, dims="minimum_category", coords={"minimum_category": MINIMUM_CATEGORIES}
    )
    temperature_difference = (challenger_temperature - reference_temperature).where(valid)
    return xarray.Dataset(
        {
            **_detection_scores(challenger_event, reference_event, prefix=""),
            **_detection_scores(
                (challenger_detection.category >= minimum_category) & valid,
                (reference_detection.category >= minimum_category) & valid,
                prefix="category_",
            ),
            "category_accuracy": _category_accuracy(
                challenger_detection.category, reference_detection.category, challenger_event | reference_event
            ),
            "temperature_difference_rmse": _root_mean_square_by_event_agreement(
                temperature_difference, challenger_event, reference_event, valid
            ),
            "temperature_bias": _weighted_mean(temperature_difference),
        }
    )


def _detection_scores(
    challenger_flag: xarray.DataArray,
    reference_flag: xarray.DataArray,
    prefix: str,
) -> dict[str, xarray.DataArray]:
    hits = _weighted_sum(challenger_flag & reference_flag)
    false_alarms = _weighted_sum(challenger_flag & ~reference_flag)
    misses = _weighted_sum(~challenger_flag & reference_flag)
    return {
        f"{prefix}probability_of_detection": _divide(hits, hits + misses),
        f"{prefix}false_alarm_ratio": _divide(false_alarms, hits + false_alarms),
        f"{prefix}critical_success_index": _divide(hits, hits + misses + false_alarms),
    }


def _category_accuracy(
    challenger_category: xarray.DataArray,
    reference_category: xarray.DataArray,
    either_event: xarray.DataArray,
) -> xarray.DataArray:
    return _divide(
        _weighted_sum((challenger_category == reference_category) & either_event), _weighted_sum(either_event)
    )


def _root_mean_square_by_event_agreement(
    difference: xarray.DataArray,
    challenger_event: xarray.DataArray,
    reference_event: xarray.DataArray,
    valid: xarray.DataArray,
) -> xarray.DataArray:
    groups = xarray.concat(
        [
            challenger_event & reference_event,
            challenger_event & ~reference_event,
            ~challenger_event & reference_event,
            ~challenger_event & ~reference_event & valid,
        ],
        dim=xarray.DataArray(EVENT_AGREEMENT_GROUPS, dims="event_agreement", name="event_agreement"),
    )
    return _weighted_mean((difference**2).where(groups)) ** 0.5


def _weighted_sum(data: xarray.DataArray) -> xarray.DataArray:
    return data.astype(float).weighted(_area_weights(data)).sum(dim=_score_dimensions(data))


def _weighted_mean(data: xarray.DataArray) -> xarray.DataArray:
    return data.weighted(_area_weights(data)).mean(dim=_score_dimensions(data))


def _area_weights(data: xarray.DataArray) -> xarray.DataArray:
    return numpy.cos(numpy.deg2rad(data[Dimension.LATITUDE.key()]))


def _score_dimensions(data: xarray.DataArray) -> list[str]:
    return [
        dimension
        for dimension in (
            Dimension.FIRST_DAY_DATETIME.key(),
            Dimension.LATITUDE.key(),
            Dimension.LONGITUDE.key(),
        )
        if dimension in data.dims
    ]


def _divide(numerator: xarray.DataArray, denominator: xarray.DataArray) -> xarray.DataArray:
    return numerator / denominator.where(denominator != 0)
