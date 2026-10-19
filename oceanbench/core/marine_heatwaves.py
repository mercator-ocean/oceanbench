# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Callable

import heatwave3
import numpy
import pandas
import xarray

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import SPATIAL_COORDINATE_ALIGNMENT_ATOL, DepthLevel, Dimension, Variable
from oceanbench.core.lead_day_utils import lead_day_labels

HISTORY_DAYS = 7
CLIMATOLOGY_DAY_OF_YEAR_DIMENSION = "doy"
CLIMATOLOGY_MEAN_VARIABLE = "seas"
CLIMATOLOGY_THRESHOLD_VARIABLE = "thresh"
MINIMUM_CATEGORIES = [2, 3, 4]
EVENT_AGREEMENT_GROUPS = ["both", "challenger_only", "reference_only", "neither"]

_DAYS_BEFORE_MONTH_IN_LEAP_YEAR = numpy.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])


def marine_heatwave_diagnostics(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    history_dataset: xarray.Dataset,
    climatology_path: str,
) -> pandas.DataFrame:
    """
    Detect marine heatwaves in challenger and reference forecast windows and score them.

    Surface temperature is taken from each dataset; the reference, the history and the
    climatology are snapped to the challenger grid by nearest index. The same reference history
    days are prepended to both windows before detection, so that an event already in progress
    at the forecast start is recognised on both sides; they are removed before scoring.

    Parameters
    ----------
    challenger_dataset, reference_dataset : xarray.Dataset
        Challenger and reference datasets with the same forecast starts and lead days.
    history_dataset : xarray.Dataset
        Reference history before each forecast start, from :func:`reference_history_dataset`.
    climatology_path : str
        Path or URL of a heatwave3 ``ts2clm3`` climatology (``seas``, ``thresh`` on ``doy``)
        covering the challenger grid.

    Returns
    -------
    pandas.DataFrame
        One row per score and one column per lead day.
    """
    challenger_temperature = _surface_temperature(challenger_dataset)
    reference_temperature = _snapped_to_challenger_grid(_surface_temperature(reference_dataset), challenger_temperature)
    history_temperature = _snapped_to_challenger_grid(
        _surface_temperature(history_dataset, lead_days=history_dataset[Dimension.LEAD_DAY_INDEX.key()].values),
        challenger_temperature,
    )
    climatology = _snapped_to_challenger_grid(_open_climatology(climatology_path), challenger_temperature)
    scores = _scores_with_history(challenger_temperature, reference_temperature, history_temperature, climatology)
    return _scores_to_dataframe(scores.compute())


def _surface_temperature(dataset: xarray.Dataset, lead_days: numpy.ndarray | None = None) -> xarray.DataArray:
    temperature = rename_dataset_with_standard_names(dataset)[Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()]
    lead_day_dimension = Dimension.LEAD_DAY_INDEX.key()
    surface_temperature = (
        temperature.sel({Dimension.DEPTH.key(): DepthLevel.SURFACE.value}, method="nearest", drop=True)
        if Dimension.DEPTH.key() in temperature.dims
        else temperature
    )
    return surface_temperature.assign_coords(
        {lead_day_dimension: numpy.arange(temperature.sizes[lead_day_dimension]) if lead_days is None else lead_days}
    )


def _snapped_to_challenger_grid(
    source: xarray.DataArray | xarray.Dataset,
    challenger_temperature: xarray.DataArray,
) -> xarray.DataArray | xarray.Dataset:
    spatial_coordinate_names = (Dimension.LATITUDE.key(), Dimension.LONGITUDE.key())
    return source.isel(
        {
            name: _nearest_indexes(source[name].values, challenger_temperature[name].values, name)
            for name in spatial_coordinate_names
        }
    ).assign_coords({name: challenger_temperature[name] for name in spatial_coordinate_names})


def _nearest_indexes(
    source_coordinate: numpy.ndarray,
    challenger_coordinate: numpy.ndarray,
    coordinate_name: str,
) -> numpy.ndarray:
    indexes = pandas.Index(source_coordinate).get_indexer(
        challenger_coordinate, method="nearest", tolerance=SPATIAL_COORDINATE_ALIGNMENT_ATOL
    )
    if (indexes < 0).any():
        raise ValueError(
            f"The {coordinate_name} coordinates do not cover the challenger grid within "
            f"{SPATIAL_COORDINATE_ALIGNMENT_ATOL} degrees: {int((indexes < 0).sum())} challenger points unmatched."
        )
    return indexes


def _open_climatology(climatology_path: str) -> xarray.Dataset:
    climatology = xarray.open_dataset(climatology_path, chunks={})
    return climatology.rename(
        {
            name: standard_name
            for name, standard_name in (("lat", Dimension.LATITUDE.key()), ("lon", Dimension.LONGITUDE.key()))
            if name in climatology.dims
        }
    )


def _scores_to_dataframe(scores: xarray.Dataset) -> pandas.DataFrame:
    rows = {
        "Marine heatwave probability of detection (-) []{surface}": scores.probability_of_detection,
        "Marine heatwave false alarm ratio (-) []{surface}": scores.false_alarm_ratio,
        "Marine heatwave critical success index (-) []{surface}": scores.critical_success_index,
        **{
            f"Marine heatwave category {roman} or above {name} (-) []{{surface}}": scores[f"category_{variable}"].sel(
                minimum_category=minimum_category
            )
            for minimum_category, roman in zip(MINIMUM_CATEGORIES, ["II", "III", "IV"])
            for variable, name in (
                ("probability_of_detection", "probability of detection"),
                ("false_alarm_ratio", "false alarm ratio"),
                ("critical_success_index", "critical success index"),
            )
        },
        "Marine heatwave category accuracy (-) []{surface}": scores.category_accuracy,
        **{
            f"Temperature RMSD on {group.replace('_', ' ')} marine heatwave days (°C) []{{surface}}": (
                scores.temperature_rmsd.sel(event_agreement=group)
            )
            for group in EVENT_AGREEMENT_GROUPS
        },
        "Temperature bias (°C) []{surface}": scores.temperature_bias,
    }
    lead_days_count = scores.sizes[Dimension.LEAD_DAY_INDEX.key()]
    return pandas.DataFrame({label: values.values for label, values in rows.items()}).set_index(
        [lead_day_labels(1, lead_days_count)]
    ).T


def _scores_with_history(
    challenger_temperature: xarray.DataArray,
    reference_temperature: xarray.DataArray,
    history_temperature: xarray.DataArray,
    climatology: xarray.Dataset,
) -> xarray.Dataset:
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
        ``temperature_rmsd`` per ``event_agreement`` group (both, challenger only,
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
            "temperature_rmsd": _root_mean_square_by_event_agreement(
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
