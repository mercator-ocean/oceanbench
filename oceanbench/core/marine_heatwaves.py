# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import xarray

from oceanbench.core.climate_forecast_standard_names import (
    rename_dataset_with_standard_names,
)
from oceanbench.core.dataset_utils import DepthLevel, Dimension, Variable
from oceanbench.core.lead_day_utils import lead_day_labels

SPATIAL_COORDINATE_ALIGNMENT_ATOL = 1e-4
SPATIAL_GRID_MINIMUM_MATCH_RATIO = 0.999
SPATIAL_COORDINATE_NAMES = (Dimension.LATITUDE.key(), Dimension.LONGITUDE.key())


def _snap_spatial_coordinates_to_challenger(
    challenger_temperature: xarray.DataArray,
    source: xarray.DataArray,
) -> xarray.DataArray:
    source_indexes_by_coordinate = {}
    challenger_coordinates = {}
    matched_grid_ratio = 1.0

    for coordinate_name in SPATIAL_COORDINATE_NAMES:
        source_index = pandas.Index(source[coordinate_name].values)
        try:
            source_indexes = source_index.get_indexer(
                challenger_temperature[coordinate_name].values,
                method="nearest",
                tolerance=SPATIAL_COORDINATE_ALIGNMENT_ATOL,
            )
        except (ValueError, pandas.errors.InvalidIndexError) as error:
            raise ValueError(
                f"Could not align {coordinate_name} coordinates: nearest-neighbor lookup failed: {error}"
            ) from error

        challenger_indexes = numpy.flatnonzero(source_indexes >= 0)
        source_indexes = source_indexes[challenger_indexes]
        if numpy.unique(source_indexes).size != source_indexes.size:
            raise ValueError(
                f"Could not align {coordinate_name} coordinates: multiple challenger coordinates match "
                f"the same source coordinate within tolerance {SPATIAL_COORDINATE_ALIGNMENT_ATOL}"
            )

        source_indexes_by_coordinate[coordinate_name] = source_indexes
        matched_grid_ratio *= challenger_indexes.size / challenger_temperature.sizes[coordinate_name]
        challenger_coordinates[coordinate_name] = challenger_temperature[coordinate_name].isel(
            {coordinate_name: challenger_indexes}
        )

    if matched_grid_ratio < SPATIAL_GRID_MINIMUM_MATCH_RATIO:
        raise ValueError(
            "Could not align spatial grid to challenger: "
            f"matched {matched_grid_ratio:.4%} of challenger grid points, "
            f"required at least {SPATIAL_GRID_MINIMUM_MATCH_RATIO:.4%}; "
            f"tolerance={SPATIAL_COORDINATE_ALIGNMENT_ATOL}"
        )

    return source.isel(source_indexes_by_coordinate).assign_coords(challenger_coordinates)


def _align_marine_heatwave_spatial_coordinates(
    challenger_temperature: xarray.DataArray,
    reference_temperature: xarray.DataArray,
    climatology_mean: xarray.DataArray,
    percentile_90: xarray.DataArray,
) -> tuple[xarray.DataArray, ...]:
    snapped_sources = (
        _snap_spatial_coordinates_to_challenger(challenger_temperature, reference_temperature),
        _snap_spatial_coordinates_to_challenger(challenger_temperature, climatology_mean),
        _snap_spatial_coordinates_to_challenger(challenger_temperature, percentile_90),
    )
    return xarray.align(
        challenger_temperature,
        *snapped_sources,
        join="inner",
        exclude={Dimension.FIRST_DAY_DATETIME.key(), Dimension.LEAD_DAY_INDEX.key()},
    )


METRIC_LABELS = {
    "probability_of_detection": "Marine heatwave probability of detection (-) []{surface}",
    "false_alarm_ratio": "Marine heatwave false alarm ratio (-) []{surface}",
    "critical_success_index": "Marine heatwave critical success index (-) []{surface}",
    "intensity_rmse": "Marine heatwave intensity RMSE (°C) []{surface}",
}


def _select_surface_temperature(dataset: xarray.Dataset) -> xarray.DataArray:
    standard_dataset = rename_dataset_with_standard_names(dataset)
    temperature = standard_dataset[Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()]

    if Dimension.DEPTH.key() in temperature.dims:
        temperature = temperature.sel({Dimension.DEPTH.key(): DepthLevel.SURFACE.value}, method="nearest")

    return temperature


def _align_climatology_to_valid_dates(
    climatology: xarray.DataArray,
    target_temperature: xarray.DataArray,
    dayofyear: xarray.DataArray | None = None,
) -> xarray.DataArray:
    if dayofyear is None:
        dayofyear = _leap_year_dayofyear(_valid_dates(target_temperature))

    if "dayofyear" in climatology.dims:
        return _select_dayofyear_climatology(climatology, dayofyear)

    if Dimension.TIME.key() in climatology.dims:
        climatology_by_day = climatology.assign_coords(
            dayofyear=_leap_year_dayofyear(climatology[Dimension.TIME.key()])
        ).swap_dims({Dimension.TIME.key(): "dayofyear"})
        return _select_dayofyear_climatology(climatology_by_day, dayofyear)

    raise ValueError("Climatology must expose a 'dayofyear' or 'time' dimension.")


def _detect_marine_heatwave_mask(
    temperature: xarray.DataArray,
    percentile_90: xarray.DataArray,
    minimum_duration: int = 5,
    allowed_gap: int = 2,
) -> xarray.DataArray:
    threshold_exceeded = temperature > percentile_90
    lead_day_dimension = Dimension.LEAD_DAY_INDEX.key()
    threshold_exceeded = threshold_exceeded.chunk({lead_day_dimension: -1})

    mask = xarray.apply_ufunc(
        _detect_marine_heatwave_events,
        threshold_exceeded,
        input_core_dims=[[lead_day_dimension]],
        output_core_dims=[[lead_day_dimension]],
        kwargs={
            "minimum_duration": minimum_duration,
            "allowed_gap": allowed_gap,
        },
        dask="parallelized",
        output_dtypes=[bool],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    return mask.transpose(*temperature.dims)


def _marine_heatwave_intensity(
    temperature: xarray.DataArray,
    climatology_mean: xarray.DataArray,
    marine_heatwave_mask: xarray.DataArray,
) -> xarray.DataArray:
    anomaly_above_climatology = temperature - climatology_mean
    return anomaly_above_climatology.where(marine_heatwave_mask, 0.0).where(temperature.notnull())


def _compute_marine_heatwave_scores(
    challenger_temperature: xarray.DataArray,
    reference_temperature: xarray.DataArray,
    climatology_mean: xarray.DataArray,
    percentile_90: xarray.DataArray,
    minimum_duration: int = 5,
    allowed_gap: int = 2,
    evaluation_lead_days: xarray.DataArray | None = None,
) -> pandas.DataFrame:
    challenger_mask = _detect_marine_heatwave_mask(
        challenger_temperature,
        percentile_90,
        minimum_duration=minimum_duration,
        allowed_gap=allowed_gap,
    )
    reference_mask = _detect_marine_heatwave_mask(
        reference_temperature,
        percentile_90,
        minimum_duration=minimum_duration,
        allowed_gap=allowed_gap,
    )

    challenger_intensity = _marine_heatwave_intensity(challenger_temperature, climatology_mean, challenger_mask)
    reference_intensity = _marine_heatwave_intensity(reference_temperature, climatology_mean, reference_mask)

    if evaluation_lead_days is not None:
        lead_day_dimension = Dimension.LEAD_DAY_INDEX.key()
        selection = {lead_day_dimension: evaluation_lead_days}
        challenger_mask = challenger_mask.sel(selection)
        reference_mask = reference_mask.sel(selection)
        challenger_intensity = challenger_intensity.sel(selection)
        reference_intensity = reference_intensity.sel(selection)

    score_dataset = xarray.Dataset(
        {
            **_event_detection_scores(challenger_mask, reference_mask),
            **_physical_scores(
                challenger_intensity=challenger_intensity,
                reference_intensity=reference_intensity,
                evaluation_mask=challenger_mask | reference_mask,
            ),
        }
    )

    return _scores_to_dataframe(score_dataset.compute())


def marine_heatwave_diagnostics(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    climatology_mean: xarray.DataArray,
    percentile_90: xarray.DataArray,
    minimum_duration: int = 5,
    allowed_gap: int = 2,
    challenger_history_dataset: xarray.Dataset | None = None,
    reference_history_dataset: xarray.Dataset | None = None,
) -> pandas.DataFrame:
    """
    Compute Marine Heatwave detection and intensity scores per forecast lead day.

    Marine heatwaves are detected on the challenger and reference surface temperatures
    following Hobday et al. (2016): the ninetieth-percentile threshold must be exceeded
    for at least ``minimum_duration`` consecutive days, and internal gaps of at most
    ``allowed_gap`` days are filled. When a challenger and reference history are provided,
    they are prepended before detection so that an event already in progress at forecast
    initialization is not treated as a new short event; the history days are excluded from
    the reported scores.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset holding surface potential temperature.
    reference_dataset : xarray.Dataset
        The reference dataset holding surface potential temperature.
    climatology_mean : xarray.DataArray
        The daily climatological mean temperature used for the Hobday intensity.
    percentile_90 : xarray.DataArray
        The daily ninetieth-percentile detection threshold.
    minimum_duration : int, optional
        The minimum number of consecutive exceedance days for an event.
    allowed_gap : int, optional
        The maximum number of internal days below the threshold merged into an event.
    challenger_history_dataset : xarray.Dataset, optional
        The challenger history prepended before detection and excluded from scores.
    reference_history_dataset : xarray.Dataset, optional
        The reference history prepended before detection and excluded from scores.

    Returns
    -------
    pandas.DataFrame
        The Marine Heatwave scores indexed by metric and one column per forecast lead day.
    """
    challenger_temperature = _select_surface_temperature(challenger_dataset)
    reference_temperature = _select_surface_temperature(reference_dataset)
    evaluation_lead_days = challenger_temperature[Dimension.LEAD_DAY_INDEX.key()]

    if (challenger_history_dataset is None) != (reference_history_dataset is None):
        raise ValueError("Challenger and reference histories must either both be provided or both be omitted.")

    if challenger_history_dataset is not None and reference_history_dataset is not None:
        challenger_history_temperature = _select_surface_temperature(challenger_history_dataset)
        reference_history_temperature = _select_surface_temperature(reference_history_dataset)
        challenger_temperature = _prepend_temperature_history(
            challenger_temperature,
            challenger_history_temperature,
        )
        reference_temperature = _prepend_temperature_history(
            reference_temperature,
            reference_history_temperature,
        )

    dayofyear = _leap_year_dayofyear(_valid_dates(challenger_temperature))
    aligned_climatology_mean = _align_climatology_to_valid_dates(climatology_mean, challenger_temperature, dayofyear)
    aligned_percentile_90 = _align_climatology_to_valid_dates(percentile_90, challenger_temperature, dayofyear)

    (
        challenger_temperature,
        reference_temperature,
        aligned_climatology_mean,
        aligned_percentile_90,
    ) = _align_marine_heatwave_spatial_coordinates(
        challenger_temperature,
        reference_temperature,
        aligned_climatology_mean,
        aligned_percentile_90,
    )
    xarray.align(
        challenger_temperature,
        reference_temperature,
        aligned_climatology_mean,
        aligned_percentile_90,
        join="exact",
    )

    return _compute_marine_heatwave_scores(
        challenger_temperature=challenger_temperature,
        reference_temperature=reference_temperature,
        climatology_mean=aligned_climatology_mean,
        percentile_90=aligned_percentile_90,
        evaluation_lead_days=evaluation_lead_days,
        minimum_duration=minimum_duration,
        allowed_gap=allowed_gap,
    )


def _prepend_temperature_history(
    forecast_temperature: xarray.DataArray,
    history_temperature: xarray.DataArray,
) -> xarray.DataArray:
    lead_day_dimension = Dimension.LEAD_DAY_INDEX.key()
    forecast_temperature, history_temperature = xarray.align(
        forecast_temperature,
        history_temperature,
        join="exact",
        exclude={lead_day_dimension},
    )

    overlapping_lead_days = numpy.intersect1d(
        forecast_temperature[lead_day_dimension],
        history_temperature[lead_day_dimension],
    )
    if overlapping_lead_days.size:
        raise ValueError("History and forecast lead-day coordinates must not overlap.")

    return xarray.concat(
        [history_temperature, forecast_temperature],
        dim=lead_day_dimension,
        join="exact",
    ).sortby(lead_day_dimension)


def _detect_marine_heatwave_events(
    threshold_exceeded: numpy.ndarray,
    minimum_duration: int,
    allowed_gap: int,
) -> numpy.ndarray:
    exceedance = numpy.asarray(threshold_exceeded, dtype=bool)
    long_enough_events = _remove_short_true_runs(exceedance, minimum_duration)
    return _fill_short_internal_false_runs(long_enough_events, allowed_gap)


def _maximal_run_bounds(
    mask: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    run_length_axis_size = mask.shape[-1]
    positions = numpy.broadcast_to(numpy.arange(run_length_axis_size), mask.shape)
    previous_outside = numpy.where(mask, -1, positions)
    run_start = numpy.maximum.accumulate(previous_outside, axis=-1) + 1
    next_outside = numpy.where(mask, run_length_axis_size, positions)
    run_stop = numpy.minimum.accumulate(next_outside[..., ::-1], axis=-1)[..., ::-1]
    return run_start, run_stop


def _remove_short_true_runs(
    mask: numpy.ndarray,
    minimum_duration: int,
) -> numpy.ndarray:
    run_start, run_stop = _maximal_run_bounds(mask)
    run_length = run_stop - run_start
    return mask & (run_length >= minimum_duration)


def _fill_short_internal_false_runs(
    mask: numpy.ndarray,
    allowed_gap: int,
) -> numpy.ndarray:
    gaps = ~mask
    run_start, run_stop = _maximal_run_bounds(gaps)
    run_length = run_stop - run_start
    is_internal_gap = (run_start > 0) & (run_stop < mask.shape[-1])
    fillable_gap = gaps & is_internal_gap & (run_length <= allowed_gap)
    return mask | fillable_gap


def _select_dayofyear_climatology(
    climatology: xarray.DataArray,
    dayofyear: xarray.DataArray,
) -> xarray.DataArray:
    return climatology.sel(dayofyear=dayofyear)


def _leap_year_dayofyear(dates: xarray.DataArray) -> xarray.DataArray:
    days_before_month = numpy.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
    leap_doy_values = days_before_month[dates.dt.month.values - 1] + dates.dt.day.values
    return xarray.DataArray(leap_doy_values, dims=dates.dims, coords=dates.coords)


def _event_detection_scores(
    challenger_mask: xarray.DataArray,
    reference_mask: xarray.DataArray,
) -> dict[str, xarray.DataArray]:
    true_positive = _weighted_sum(challenger_mask & reference_mask)
    false_positive = _weighted_sum(challenger_mask & ~reference_mask)
    false_negative = _weighted_sum(~challenger_mask & reference_mask)

    return {
        "probability_of_detection": _safe_divide(true_positive, true_positive + false_negative),
        "false_alarm_ratio": _safe_divide(false_positive, true_positive + false_positive),
        "critical_success_index": _safe_divide(
            true_positive,
            true_positive + false_positive + false_negative,
        ),
    }


def _physical_scores(
    challenger_intensity: xarray.DataArray,
    reference_intensity: xarray.DataArray,
    evaluation_mask: xarray.DataArray,
) -> dict[str, xarray.DataArray]:
    intensity_squared_error = (challenger_intensity - reference_intensity) ** 2
    return {
        "intensity_rmse": _weighted_mean(intensity_squared_error.where(evaluation_mask)) ** 0.5,
    }


def _weighted_sum(data: xarray.DataArray) -> xarray.DataArray:
    return data.astype(float).weighted(_spatial_area_weights(data)).sum(dim=_score_dimensions(data))


def _weighted_mean(data: xarray.DataArray) -> xarray.DataArray:
    return data.astype(float).weighted(_spatial_area_weights(data)).mean(dim=_score_dimensions(data))


def _spatial_area_weights(data: xarray.DataArray) -> xarray.DataArray:
    return numpy.cos(numpy.deg2rad(data[Dimension.LATITUDE.key()]))


def _score_dimensions(data: xarray.DataArray) -> list[str]:
    return [dimension for dimension in data.dims if dimension != Dimension.LEAD_DAY_INDEX.key()]


def _safe_divide(numerator: xarray.DataArray, denominator: xarray.DataArray) -> xarray.DataArray:
    return numerator / denominator.where(denominator != 0)


def _valid_dates(data: xarray.DataArray) -> xarray.DataArray:
    first_days = data[Dimension.FIRST_DAY_DATETIME.key()]
    lead_days = data[Dimension.LEAD_DAY_INDEX.key()]
    return first_days + lead_days.astype("timedelta64[D]")


def _scores_to_dataframe(score_dataset: xarray.Dataset) -> pandas.DataFrame:
    lead_days_count = score_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    values = {METRIC_LABELS[name]: score_dataset[name].values for name in score_dataset.data_vars}
    return pandas.DataFrame(values).set_index([lead_day_labels(1, lead_days_count)]).T
