# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Smoothing-controlled RMSD against a gridded reference (GLORYS reanalysis or GLO12 analysis).

The gridded counterpart of the Class IV version: the forecast anomaly against a monthly climatology is smoothed by
Gaussians of the same widths in km, rescaled through the origin, and the width and slope are cross fitted on odd and
even initialization days. Masks, depths, lead days and area weights are those of the plain gridded RMSD.
"""

import numpy
import pandas
import xarray

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, DepthLevel, Variable
from oceanbench.core.rmsd import (
    DEPTH_LABELS,
    MISSING_FRACTION_COLUMN,
    _assign_depth_dimension,
    _harmonise_dataset,
    _has_depths,
    _missing_fractions,
    _ocean_mask_on_challenger_grid,
    _select_variables,
    _snap_reference_spatial_coordinates_to_challenger,
    _to_pretty_dataframe,
    _variable_ocean_mask,
)
from oceanbench.core.smoothing_controlled_rmsd import (
    ROW_DIAGNOSTICS_ATTRIBUTE,
    SMOOTHING_SLOPE_ROW_FIELD,
    SMOOTHING_WIDTH_ROW_FIELD,
    SMOOTHING_WIDTHS_KILOMETERS,
    SUFFICIENT_STATISTICS_ATTRIBUTE,
    _monthly_interpolation,
    grid_smoothed_fields,
    score_smoothing_controlled_rmsd,
)


def _climatology_on_challenger_grid(
    climatology_dataset: xarray.Dataset,
    challenger_dataset: xarray.Dataset,
    variables: list[Variable],
) -> xarray.Dataset:
    climatology = _select_variables(rename_dataset_with_standard_names(climatology_dataset), variables)
    time_key = Dimension.TIME.key()
    month_order = numpy.argsort(pandas.DatetimeIndex(climatology[time_key].values).month.values)
    climatology = climatology.isel({time_key: month_order})
    if Dimension.DEPTH.key() in climatology.dims:
        climatology = _assign_depth_dimension(
            climatology.sel({Dimension.DEPTH.key(): [level.value for level in DepthLevel]}, method="nearest")
        )
    return _snap_reference_spatial_coordinates_to_challenger(challenger_dataset, climatology)


def _ordered_spatial_values(field: xarray.DataArray) -> numpy.ndarray:
    latitude_key, longitude_key = Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()
    other_dimensions = [dimension for dimension in field.dims if dimension not in (latitude_key, longitude_key)]
    return numpy.asarray(field.transpose(*other_dimensions, latitude_key, longitude_key).values, dtype=float)


def _climatology_at_date(
    climatology_variable: xarray.DataArray,
    date: numpy.datetime64,
    month_cache: dict,
) -> numpy.ndarray:
    """Monthly climatology linear in time between mid months; ``month_cache`` keeps the months in use only."""
    lower_months, upper_months, upper_weights = _monthly_interpolation(numpy.array([date]))
    needed_months = {int(lower_months[0]), int(upper_months[0])}
    for cached_month in [cached_month for cached_month in month_cache if cached_month not in needed_months]:
        del month_cache[cached_month]
    for month in needed_months - set(month_cache):
        month_cache[month] = _ordered_spatial_values(climatology_variable.isel({Dimension.TIME.key(): month}).compute())
    lower, upper = month_cache[int(lower_months[0])], month_cache[int(upper_months[0])]
    return lower + upper_weights[0] * (upper - lower)


def _sufficient_statistics_of_field(
    forecast_anomaly: numpy.ndarray,
    reference_anomaly: numpy.ndarray,
    area_weights: numpy.ndarray,
    template: xarray.DataArray,
) -> dict[str, numpy.ndarray]:
    """
    Weighted sums over the grid, shape (widths,) + leading axes of the field.

    ``forecast_anomaly`` is NaN on land. The smoothed anomaly falls back to the point anomaly where it has no ocean
    within reach, as in the Class IV version.
    """
    smoothed = numpy.stack(
        [smoothed_field.values for smoothed_field in grid_smoothed_fields(template.copy(data=forecast_anomaly))]
    )
    anomalies = numpy.concatenate([forecast_anomaly[None], smoothed])
    anomalies = numpy.where(numpy.isfinite(anomalies), anomalies, forecast_anomaly[None])
    is_scored = numpy.isfinite(forecast_anomaly) & numpy.isfinite(reference_anomaly)
    weights = numpy.where(is_scored, area_weights, 0.0)
    anomalies = numpy.where(is_scored[None], anomalies, 0.0)
    references = numpy.where(is_scored, reference_anomaly, 0.0)
    spatial_axes = (-2, -1)
    return {
        "count": numpy.broadcast_to(weights.sum(axis=spatial_axes), anomalies.shape[:-2]),
        "sum_yy": numpy.broadcast_to((weights * references**2).sum(axis=spatial_axes), anomalies.shape[:-2]),
        "sum_aa": (weights * anomalies**2).sum(axis=spatial_axes),
        "sum_ya": (weights * references * anomalies).sum(axis=spatial_axes),
    }


def _statistics_records(
    statistics: dict[str, numpy.ndarray],
    depth_labels: list[str],
    first_day: pandas.Timestamp,
    lead_day: int,
) -> pandas.DataFrame:
    widths_count = len(SMOOTHING_WIDTHS_KILOMETERS)
    return pandas.DataFrame(
        {
            "first_day": first_day,
            "depth_bin": numpy.tile(depth_labels, widths_count),
            "lead_day": lead_day,
            "width_km": numpy.repeat(SMOOTHING_WIDTHS_KILOMETERS, len(depth_labels)),
        }
        | {name: values.reshape(widths_count, len(depth_labels)).ravel() for name, values in statistics.items()}
    )


def gridded_sufficient_statistics(
    challenger_variable: xarray.DataArray,
    reference_variable: xarray.DataArray,
    climatology_variable: xarray.DataArray,
    ocean_mask: xarray.DataArray,
    scored_cells: numpy.ndarray | None = None,
) -> pandas.DataFrame:
    """
    Per initialization day, depth, lead day and smoothing width weighted sums of one variable.

    All inputs are on the challenger grid, the challenger and reference with the first day, lead day, (depth),
    latitude and longitude dimensions of the harmonised plain RMSD datasets. One lead day of one initialization day is
    held in memory at a time. ``scored_cells`` (latitude, longitude booleans) restricts the scored cells, the
    smoothing still sees the whole challenger.
    """
    latitude_key, longitude_key = Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()
    first_day_key, lead_day_key = Dimension.FIRST_DAY_DATETIME.key(), Dimension.LEAD_DAY_INDEX.key()
    has_depth = Dimension.DEPTH.key() in challenger_variable.dims
    depth_labels = list(challenger_variable[Dimension.DEPTH.key()].values) if has_depth else ["surface"]
    is_ocean = _ordered_spatial_values(ocean_mask).astype(bool)
    latitudes = challenger_variable[latitude_key].values
    area_weights = numpy.broadcast_to(numpy.cos(numpy.deg2rad(latitudes))[:, None], is_ocean.shape[-2:])
    if scored_cells is not None:
        area_weights = numpy.where(scored_cells, area_weights, 0.0)
    month_cache: dict = {}
    records = []
    for first_day in numpy.sort(challenger_variable[first_day_key].values):
        for lead_day in challenger_variable[lead_day_key].values:
            selection = {first_day_key: first_day, lead_day_key: lead_day}
            forecast_field = challenger_variable.sel(selection).compute()
            template = forecast_field.transpose(
                *[dimension for dimension in forecast_field.dims if dimension not in (latitude_key, longitude_key)],
                latitude_key,
                longitude_key,
            )
            climatology = _climatology_at_date(
                climatology_variable, first_day + numpy.timedelta64(int(lead_day), "D"), month_cache
            )
            forecast_anomaly = numpy.where(is_ocean, _ordered_spatial_values(forecast_field) - climatology, numpy.nan)
            reference_anomaly = numpy.where(
                is_ocean, _ordered_spatial_values(reference_variable.sel(selection).compute()) - climatology, numpy.nan
            )
            statistics = _sufficient_statistics_of_field(forecast_anomaly, reference_anomaly, area_weights, template)
            if not has_depth:
                statistics = {name: values[:, None] for name, values in statistics.items()}
            records.append(_statistics_records(statistics, depth_labels, pandas.Timestamp(first_day), int(lead_day)))
    return pandas.concat(records, ignore_index=True)


def _scores_as_dataarray(scores: pandas.DataFrame, value_column: str, has_depth: bool) -> xarray.DataArray:
    table = scores.pivot(index="lead_day", columns="depth_bin", values=value_column).sort_index()
    if not has_depth:
        return xarray.DataArray(table.iloc[:, 0].values, dims=[Dimension.LEAD_DAY_INDEX.key()])
    depth_labels = [label for label in DEPTH_LABELS.values() if label in table.columns]
    return xarray.DataArray(
        table[depth_labels].values,
        dims=[Dimension.LEAD_DAY_INDEX.key(), Dimension.DEPTH.key()],
        coords={Dimension.DEPTH.key(): depth_labels},
    )


def smoothing_controlled_rmsd_gridded(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
    ocean_mask: xarray.DataArray,
    climatology_dataset: xarray.Dataset,
    scored_extent: tuple[float, float, float, float] | None = None,
) -> pandas.DataFrame:
    """
    Smoothing-controlled RMSD table against a gridded reference, shaped like the plain gridded RMSD table.

    ``scored_extent`` (minimum and maximum latitude and longitude): when the challenger extends beyond the evaluated
    region so that it can be smoothed there, only the cells within the extent are scored.
    The fitted width and slope of each cell are in ``table.attrs["row_diagnostics"]`` and the per initialization day
    sufficient statistics of each variable in ``table.attrs["sufficient_statistics"]``.
    """
    challenger = _select_variables(_harmonise_dataset(challenger_dataset), variables)
    reference = _snap_reference_spatial_coordinates_to_challenger(
        challenger, _select_variables(_harmonise_dataset(reference_dataset), variables)
    )
    climatology = _climatology_on_challenger_grid(climatology_dataset, challenger, variables)
    challenger_ocean_mask = _ocean_mask_on_challenger_grid(ocean_mask, challenger)
    scored_cells = None
    if scored_extent is not None:
        minimum_latitude, maximum_latitude, minimum_longitude, maximum_longitude = scored_extent
        latitudes = challenger[Dimension.LATITUDE.key()].values[:, None]
        longitudes = challenger[Dimension.LONGITUDE.key()].values[None, :]
        scored_cells = (
            (latitudes >= minimum_latitude)
            & (latitudes <= maximum_latitude)
            & (longitudes >= minimum_longitude)
            & (longitudes <= maximum_longitude)
        )
    all_statistics = {}
    score_arrays: dict[str, dict[str, xarray.DataArray]] = {"rmsd": {}, "width_km": {}, "slope": {}}
    for variable in variables:
        variable_key = variable.key()
        statistics = gridded_sufficient_statistics(
            challenger[variable_key],
            reference[variable_key],
            climatology[variable_key],
            _variable_ocean_mask(challenger_ocean_mask, challenger, variable_key),
            scored_cells,
        )
        all_statistics[variable_key] = statistics
        scores = score_smoothing_controlled_rmsd(statistics)
        for value_column, arrays in score_arrays.items():
            arrays[variable_key] = _scores_as_dataarray(scores, value_column, _has_depths(challenger, variable_key))
    missing_fraction_dataset = _missing_fractions(challenger, reference, challenger_ocean_mask).compute()
    formatted = {
        value_column: _to_pretty_dataframe(xarray.Dataset(arrays), variables, missing_fraction_dataset)
        for value_column, arrays in score_arrays.items()
    }
    result = formatted["rmsd"]
    result.attrs[ROW_DIAGNOSTICS_ATTRIBUTE] = {
        SMOOTHING_WIDTH_ROW_FIELD: formatted["width_km"].drop(columns=[MISSING_FRACTION_COLUMN]),
        SMOOTHING_SLOPE_ROW_FIELD: formatted["slope"].drop(columns=[MISSING_FRACTION_COLUMN]),
    }
    result.attrs[SUFFICIENT_STATISTICS_ATTRIBUTE] = all_statistics
    return result
