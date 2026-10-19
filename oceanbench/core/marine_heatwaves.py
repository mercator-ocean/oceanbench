# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Callable

import heatwave3
import numpy
import xarray

from oceanbench.core.dataset_utils import DepthLevel, Dimension

HISTORY_DAYS = 7
CLIMATOLOGY_DAY_OF_YEAR_DIMENSION = "doy"
CLIMATOLOGY_MEAN_VARIABLE = "seas"
CLIMATOLOGY_THRESHOLD_VARIABLE = "thresh"
MINIMUM_CATEGORIES = [2, 3, 4]
EVENT_AGREEMENT_GROUPS = ["both", "challenger_only", "reference_only", "neither"]

_DAYS_BEFORE_MONTH_IN_LEAP_YEAR = numpy.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])


def marine_heatwave_diagnostics(
    challenger_temperature: xarray.DataArray,
    reference_temperature: xarray.DataArray,
    history_temperature: xarray.DataArray,
    climatology: xarray.Dataset,
) -> xarray.Dataset:
    """
    Detect marine heatwaves in challenger and reference forecast windows and score them.

    The same history days are prepended to both windows before detection, so that an event
    already in progress at the forecast start is recognised on both sides; they are removed
    before scoring.

    Parameters
    ----------
    challenger_temperature, reference_temperature : xarray.DataArray
        Daily surface temperature on the same grid, forecast starts and lead days (from 0).
    history_temperature : xarray.DataArray
        Reference surface temperature on negative lead days before each forecast start, from
        :func:`reference_history_dataset`.
    climatology : xarray.Dataset
        heatwave3 climatology (``seas``, ``thresh`` on ``doy``) on the same grid.

    Returns
    -------
    xarray.Dataset
        The scores of :func:`marine_heatwave_scores` for the forecast lead days.
    """
    forecast_lead_days = challenger_temperature[Dimension.LEAD_DAY_INDEX.key()]
    challenger_detection = detect_marine_heatwaves(
        _prepend_history(challenger_temperature, history_temperature), climatology
    )
    reference_detection = detect_marine_heatwaves(
        _prepend_history(reference_temperature, history_temperature), climatology
    )
    return marine_heatwave_scores(
        challenger_temperature,
        reference_temperature,
        challenger_detection.sel({Dimension.LEAD_DAY_INDEX.key(): forecast_lead_days}),
        reference_detection.sel({Dimension.LEAD_DAY_INDEX.key(): forecast_lead_days}),
    )


def reference_history_dataset(
    challenger_dataset: xarray.Dataset,
    reference_loader: Callable[[xarray.Dataset], xarray.Dataset],
    history_days: int = HISTORY_DAYS,
) -> xarray.Dataset:
    """
    Load the reference surface fields of the days before every forecast start.

    The reference loader is asked for a window of ``history_days`` days ending the day before
    each forecast start, at the surface only, and the result is relabelled to the forecast
    starts with lead days ``-history_days`` to ``-1``.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset, for its forecast starts, grid and depth coordinate.
    reference_loader : Callable[[xarray.Dataset], xarray.Dataset]
        A reference opener taking a challenger-like dataset, such as the GLORYS or GLO12 one.
    history_days : int, optional
        Number of days before each forecast start.

    Returns
    -------
    xarray.Dataset
        The reference history with the challenger's forecast starts and negative lead days.
    """
    first_day_dimension = Dimension.FIRST_DAY_DATETIME.key()
    lead_day_dimension = Dimension.LEAD_DAY_INDEX.key()
    forecast_first_days = challenger_dataset[first_day_dimension]
    history_request = (
        challenger_dataset.isel({lead_day_dimension: slice(0, 1)})
        .sel({Dimension.DEPTH.key(): [DepthLevel.SURFACE.value]}, method="nearest")
        .reindex({lead_day_dimension: numpy.arange(history_days)})
        .assign_coords({first_day_dimension: forecast_first_days - numpy.timedelta64(history_days, "D")})
    )
    return reference_loader(history_request).assign_coords(
        {
            first_day_dimension: forecast_first_days.values,
            lead_day_dimension: numpy.arange(-history_days, 0),
        }
    )


def _prepend_history(temperature: xarray.DataArray, history_temperature: xarray.DataArray) -> xarray.DataArray:
    return xarray.concat(
        [history_temperature, temperature],
        dim=Dimension.LEAD_DAY_INDEX.key(),
        join="exact",
        coords="minimal",
        compat="override",
    )


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
