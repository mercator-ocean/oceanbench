# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Smoothing-controlled RMSD against Class IV observations.

The RMSD a challenger reaches after the best Gaussian smoothing and amplitude rescaling of its anomaly
against a climatology, fitted on half of the forecast initialization days and scored on the other half.
"""

import numpy
import pandas
import xarray

from oceanbench.core.classIV_support import (
    _horizontally_interpolated_profiles,
    _interpolate_vertically_bracket,
    _interpolated_model_values_for_observation_group,
    _mean_dynamic_topography_on_challenger_grid,
    _model_data_with_depth_dimension,
    class4_observations_in_shared_population,
    class4_population_layers,
    create_class4_observations_dataframe,
    format_class4_results,
    prepare_class4_model_variable,
)
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.references.observations import load_mean_dynamic_topography
from oceanbench.core.resolution import get_dataset_resolution

SIGMAS_KILOMETERS = (15.0, 30.0, 45.0, 60.0, 90.0)
SMOOTHING_WIDTHS_KILOMETERS = (0.0,) + SIGMAS_KILOMETERS
KILOMETERS_PER_DEGREE = 111.195
# To be unified with the constant of the same name introduced by pull request #393 at rebase.
GLORYS_MEAN_SEA_SURFACE_HEIGHT_SHIFT = -0.0646
SMOOTHING_WIDTH_ROW_FIELD = "smoothing_width_km"
SMOOTHING_SLOPE_ROW_FIELD = "smoothing_slope"
ROW_DIAGNOSTICS_ATTRIBUTE = "row_diagnostics"
_OBSERVATION_CHUNK_SIZE = 2000
_MINIMUM_RENORMALISATION_WEIGHT = 1e-6

# Polar rings of the stencil around a centre point: (radius in km, point count), odd rings rotated by half a step.
# Weights fitted by minimax so the plane wave response is within 0.05 of the Gaussian one for wavelengths >= 20 km.
_STENCIL_RINGS = (
    (16.0, 10),
    (32.6, 15),
    (49.1, 20),
    (65.7, 25),
    (82.2, 30),
    (98.8, 36),
    (115.3, 41),
    (131.9, 46),
    (148.4, 51),
    (165.0, 56),
)
# One row per sigma of SIGMAS_KILOMETERS, one column per ring (centre first), each summing to one.
_STENCIL_RING_WEIGHTS = numpy.array(
    [
        [0.160426, 0.548617, 0.228150, 0.032108, 0.019437, 0.0, 0.008297, 0.0, 0.0, 0.0, 0.002965],
        [0.065805, 0.206289, 0.287471, 0.221507, 0.133005, 0.060517, 0.016480, 0.0, 0.0, 0.002802, 0.006124],
        [0.029234, 0.097566, 0.183412, 0.215102, 0.197385, 0.140648, 0.076501, 0.028779, 0.006110, 0.011527, 0.013736],
        [0.016453, 0.057656, 0.124604, 0.160794, 0.180240, 0.159061, 0.123041, 0.081298, 0.038308, 0.034094, 0.024452],
        [0.003145, 0.031256, 0.063601, 0.086910, 0.105835, 0.111293, 0.114948, 0.101681, 0.101342, 0.016207, 0.263781],
    ]
)

STATISTICS_COLUMNS = ["count", "sum_yy", "sum_aa", "sum_ya"]


def stencil_offsets_kilometers() -> numpy.ndarray:
    """Return the eastward and northward offsets of the stencil points in km, shape (points, 2)."""
    rings = [numpy.zeros((1, 2))] + [
        radius
        * numpy.stack(
            [
                numpy.cos(numpy.pi / count * (ring_index % 2) + 2 * numpy.pi * numpy.arange(count) / count),
                numpy.sin(numpy.pi / count * (ring_index % 2) + 2 * numpy.pi * numpy.arange(count) / count),
            ],
            axis=1,
        )
        for ring_index, (radius, count) in enumerate(_STENCIL_RINGS)
    ]
    return numpy.concatenate(rings)


def stencil_weights() -> numpy.ndarray:
    """Return the weight of each stencil point for each sigma, shape (sigmas, points), rows summing to one."""
    point_counts = numpy.array([1] + [count for _, count in _STENCIL_RINGS])
    ring_weights = _STENCIL_RING_WEIGHTS / _STENCIL_RING_WEIGHTS.sum(axis=1, keepdims=True)
    return numpy.repeat(ring_weights / point_counts, point_counts, axis=1)


def stencil_response(wavelength_kilometers: numpy.ndarray, direction_radians: numpy.ndarray) -> numpy.ndarray:
    """Return the stencil response to plane waves, shape (sigmas, wavelengths, directions)."""
    offsets = stencil_offsets_kilometers()
    wavenumbers = 2 * numpy.pi / numpy.asarray(wavelength_kilometers, dtype=float)
    projections = (
        numpy.cos(direction_radians)[None, :, None] * offsets[None, None, :, 0]
        + numpy.sin(direction_radians)[None, :, None] * offsets[None, None, :, 1]
    )
    cosines = numpy.cos(wavenumbers[:, None, None] * projections)
    return numpy.einsum("sk,ldk->sld", stencil_weights(), cosines)


def stencil_positions(
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    offsets = stencil_offsets_kilometers()
    stencil_latitudes = latitudes[:, None] + offsets[None, :, 1] / KILOMETERS_PER_DEGREE
    stencil_longitudes = longitudes[:, None] + offsets[None, :, 0] / (
        KILOMETERS_PER_DEGREE * numpy.cos(numpy.deg2rad(latitudes))[:, None]
    )
    return stencil_latitudes, stencil_longitudes


def _gaussian_means_of_chunk(
    time_slice: xarray.DataArray,
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
    weights: numpy.ndarray,
) -> numpy.ndarray:
    stencil_latitudes, stencil_longitudes = stencil_positions(latitudes, longitudes)
    points_count = stencil_latitudes.shape[1]
    samples = _horizontally_interpolated_profiles(
        time_slice,
        pandas.DataFrame(
            {
                Dimension.LATITUDE.key(): stencil_latitudes.ravel(),
                Dimension.LONGITUDE.key(): stencil_longitudes.ravel(),
            }
        ),
    ).reshape(-1, len(latitudes), points_count)
    is_finite = numpy.isfinite(samples)
    weighted_sums = numpy.einsum("sk,dnk->sdn", weights, numpy.where(is_finite, samples, 0.0))
    total_weights = numpy.einsum("sk,dnk->sdn", weights, is_finite.astype(float))
    return numpy.where(
        total_weights > _MINIMUM_RENORMALISATION_WEIGHT,
        weighted_sums / numpy.maximum(total_weights, _MINIMUM_RENORMALISATION_WEIGHT),
        numpy.nan,
    )


def gaussian_means_at_observations(
    time_slice: xarray.DataArray,
    observation_group: pandas.DataFrame,
    model_depths: numpy.ndarray,
) -> numpy.ndarray:
    """
    Gaussian weighted means of a model field around each observation, one row per sigma.

    The field is sampled by the Class IV bilinear interpolation at the stencil points, land samples are
    skipped and the weights renormalised, on each model level, then interpolated vertically like the point value.
    """
    weights = stencil_weights()
    latitudes = observation_group[Dimension.LATITUDE.key()].values
    longitudes = observation_group[Dimension.LONGITUDE.key()].values
    observation_depths = observation_group[Dimension.DEPTH.key()].values
    level_means = numpy.concatenate(
        [
            _gaussian_means_of_chunk(time_slice, latitudes[start:stop], longitudes[start:stop], weights)
            for start, stop in _chunk_bounds(len(observation_group))
        ],
        axis=2,
    )
    return numpy.stack(
        [_interpolate_vertically_bracket(sigma_means, model_depths, observation_depths) for sigma_means in level_means]
    )


def _chunk_bounds(count: int) -> list[tuple[int, int]]:
    return [(start, min(start + _OBSERVATION_CHUNK_SIZE, count)) for start in range(0, count, _OBSERVATION_CHUNK_SIZE)]


def _mid_month_days(year: int) -> numpy.ndarray:
    month_starts = pandas.date_range(f"{year - 1}-12-01", f"{year + 1}-02-01", freq="MS")
    return (month_starts[:-1] + (month_starts[1:] - month_starts[:-1]) / 2).values


def _monthly_interpolation(times: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    timestamps = pandas.DatetimeIndex(times)
    lower_months = numpy.empty(len(timestamps), dtype=int)
    upper_months = numpy.empty(len(timestamps), dtype=int)
    upper_weights = numpy.empty(len(timestamps), dtype=float)
    for year in numpy.unique(timestamps.year):
        is_year = timestamps.year == year
        mid_months = _mid_month_days(year)
        positions = numpy.searchsorted(mid_months, timestamps[is_year].values, side="right") - 1
        lower_months[is_year] = (positions - 1) % 12
        upper_months[is_year] = positions % 12
        upper_weights[is_year] = (timestamps[is_year].values - mid_months[positions]) / (
            mid_months[positions + 1] - mid_months[positions]
        )
    return lower_months, upper_months, upper_weights


def climatology_at_observations(
    climatology_variable: xarray.DataArray,
    observations_dataframe: pandas.DataFrame,
) -> numpy.ndarray:
    """
    Monthly climatology at each observation, with the Class IV interpolation, linear in time between mid months.

    The climatology has one time step per calendar month.
    """
    time_key = Dimension.TIME.key()
    climatology_variable = _model_data_with_depth_dimension(climatology_variable)
    month_order = numpy.argsort(pandas.DatetimeIndex(climatology_variable[time_key].values).month.values)
    climatology_variable = climatology_variable.isel({time_key: month_order})
    climatology_depths = climatology_variable[Dimension.DEPTH.key()].values
    observation_depths = observations_dataframe[Dimension.DEPTH.key()].values
    lower_months, upper_months, upper_weights = _monthly_interpolation(observations_dataframe[time_key].values)
    values = numpy.full(len(observations_dataframe), numpy.nan)
    for start, stop in _chunk_bounds(len(observations_dataframe)):
        profiles = _horizontally_interpolated_profiles(climatology_variable, observations_dataframe.iloc[start:stop])
        monthly_values = numpy.stack(
            [
                _interpolate_vertically_bracket(month_profiles, climatology_depths, observation_depths[start:stop])
                for month_profiles in profiles
            ]
        )
        observation_indices = numpy.arange(stop - start)
        lower_values = monthly_values[lower_months[start:stop], observation_indices]
        upper_values = monthly_values[upper_months[start:stop], observation_indices]
        values[start:stop] = lower_values + upper_weights[start:stop] * (upper_values - lower_values)
    return values


def prepare_climatology_variable(
    climatology_variable: xarray.DataArray,
    variable_key: str,
) -> xarray.DataArray:
    """Convert the climatology sea surface height to a sea level anomaly as the Class IV does for GLORYS."""
    if variable_key != Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key():
        return climatology_variable
    resolution = get_dataset_resolution(climatology_variable.to_dataset(name="__resolution__"))
    mean_dynamic_topography = _mean_dynamic_topography_on_challenger_grid(
        load_mean_dynamic_topography(resolution),
        climatology_variable,
    )
    return climatology_variable - mean_dynamic_topography - GLORYS_MEAN_SEA_SURFACE_HEIGHT_SHIFT


def _sufficient_statistics_of_group(
    observations: numpy.ndarray,
    point_values: numpy.ndarray,
    gaussian_means: numpy.ndarray,
    climatology_values: numpy.ndarray,
) -> dict[str, numpy.ndarray]:
    forecasts = numpy.vstack([point_values[None, :], gaussian_means])
    is_scored = numpy.isfinite(point_values) & numpy.isfinite(climatology_values)
    observation_anomalies = (observations - climatology_values)[is_scored]
    forecast_anomalies = numpy.where(
        numpy.isfinite(forecasts[:, is_scored]), forecasts[:, is_scored], point_values[is_scored]
    ) - (climatology_values[is_scored])
    return {
        "count": numpy.full(len(SMOOTHING_WIDTHS_KILOMETERS), is_scored.sum()),
        "sum_yy": numpy.full(len(SMOOTHING_WIDTHS_KILOMETERS), numpy.sum(observation_anomalies**2)),
        "sum_aa": numpy.sum(forecast_anomalies**2, axis=1),
        "sum_ya": forecast_anomalies @ observation_anomalies,
    }


def sufficient_statistics_for_first_day(
    model_data: xarray.DataArray,
    climatology_values: numpy.ndarray,
    first_day_group: pandas.DataFrame,
    first_day_index: int,
    lead_day_to_index: dict[object, int],
) -> pandas.DataFrame:
    """
    Per depth bin, lead day and smoothing width sufficient statistics of one forecast initialization day.

    ``climatology_values`` is aligned with ``first_day_group``. Observations whose point value or climatology
    is missing are not counted, as the plain RMSD leaves out observations without a point value.
    """
    model_depths = model_data[Dimension.DEPTH.key()].values
    first_day_block = model_data.isel({Dimension.FIRST_DAY_DATETIME.key(): first_day_index}).compute()
    records = []
    group_climatology = pandas.Series(climatology_values, index=first_day_group.index)
    for (lead_day, depth_bin), observation_group in first_day_group.groupby(["lead_day", "depth_bin"], sort=False):
        time_slice = first_day_block.isel({Dimension.LEAD_DAY_INDEX.key(): lead_day_to_index[lead_day]})
        statistics = _sufficient_statistics_of_group(
            observation_group["observation_value"].values,
            _interpolated_model_values_for_observation_group(time_slice, observation_group, model_depths),
            gaussian_means_at_observations(time_slice, observation_group, model_depths),
            group_climatology.loc[observation_group.index].values,
        )
        records.append(
            pandas.DataFrame(
                {
                    "depth_bin": depth_bin,
                    "lead_day": lead_day,
                    "width_km": SMOOTHING_WIDTHS_KILOMETERS,
                    "total": len(observation_group),
                }
                | statistics
            )
        )
    return pandas.concat(records, ignore_index=True)


def _fold_fit(statistics: pandas.DataFrame) -> tuple[float, float]:
    sums = statistics.groupby("width_km")[STATISTICS_COLUMNS].sum()
    slopes = numpy.where(sums["sum_aa"] > 0, sums["sum_ya"] / sums["sum_aa"].where(sums["sum_aa"] > 0, 1.0), 0.0)
    errors = sums["sum_yy"] - 2 * slopes * sums["sum_ya"] + slopes**2 * sums["sum_aa"]
    best = int(numpy.argmin(errors.values))
    return float(sums.index[best]), float(slopes[best])


def sum_of_squared_errors(statistics: pandas.DataFrame, width_km: float, slope: float) -> float:
    """Sum over starts of the squared error of ``slope`` times the forecast anomaly smoothed at ``width_km``."""
    sums = statistics.loc[statistics["width_km"] == width_km, STATISTICS_COLUMNS].sum()
    return float(sums["sum_yy"] - 2 * slope * sums["sum_ya"] + slope**2 * sums["sum_aa"])


def _first_day_folds(first_days: numpy.ndarray) -> dict[object, int]:
    return {first_day: index % 2 for index, first_day in enumerate(numpy.sort(numpy.unique(first_days)))}


def _score_group(statistics: pandas.DataFrame, folds: dict[object, int]) -> dict[str, float]:
    count = statistics.loc[statistics["width_km"] == 0.0, "count"].sum()
    fold_of_rows = statistics["first_day"].map(folds)
    fold_statistics = [statistics.loc[fold_of_rows == fold] for fold in (0, 1)]
    has_both_folds = all((part.loc[part["width_km"] == 0.0, "count"].sum() > 0) for part in fold_statistics)
    width_km, slope = _fold_fit(statistics)
    if has_both_folds:
        error = sum(
            sum_of_squared_errors(fold_statistics[1 - fold], *_fold_fit(fold_statistics[fold])) for fold in (0, 1)
        )
    else:
        error = sum_of_squared_errors(statistics, width_km, slope)
    return {
        "rmsd": numpy.sqrt(max(error, 0.0) / count) if count > 0 else numpy.nan,
        "width_km": width_km if count > 0 else numpy.nan,
        "slope": slope if count > 0 else numpy.nan,
        "cross_fitted": bool(has_both_folds),
    }


def score_smoothing_controlled_rmsd(statistics: pandas.DataFrame) -> pandas.DataFrame:
    """
    Cross fitted smoothing-controlled RMSD from per start sufficient statistics.

    ``statistics`` has one row per ``first_day``, ``depth_bin``, ``lead_day`` and ``width_km`` with the columns
    ``count``, ``sum_yy``, ``sum_aa`` and ``sum_ya``. Start dates are sorted and split into odd and even ranks; the
    width and slope fitted on one fold score the other. A depth bin and lead day seen on one fold only is fitted
    and scored in sample (``cross_fitted`` is false). ``width_km`` and ``slope`` are fitted on all starts.
    """
    folds = _first_day_folds(statistics["first_day"].values)
    records = [
        {"depth_bin": depth_bin, "lead_day": lead_day} | _score_group(group, folds)
        for (depth_bin, lead_day), group in statistics.groupby(["depth_bin", "lead_day"], sort=False)
    ]
    return pandas.DataFrame.from_records(
        records, columns=["depth_bin", "lead_day", "rmsd", "width_km", "slope", "cross_fitted"]
    )


def _variable_sufficient_statistics(
    model_variable: xarray.DataArray,
    climatology_variable: xarray.DataArray,
    observations_dataframe: pandas.DataFrame,
) -> pandas.DataFrame:
    observations_dataframe = observations_dataframe.reset_index(drop=True)
    model_data = _model_data_with_depth_dimension(model_variable)
    first_days = model_data[Dimension.FIRST_DAY_DATETIME.key()].values
    first_day_to_index = {pandas.Timestamp(first_day): index for index, first_day in enumerate(first_days)}
    lead_day_to_index = {
        lead_day: index for index, lead_day in enumerate(model_data[Dimension.LEAD_DAY_INDEX.key()].values)
    }
    return pandas.concat(
        [
            sufficient_statistics_for_first_day(
                model_data,
                climatology_at_observations(climatology_variable, first_day_group),
                first_day_group,
                first_day_to_index[pandas.Timestamp(first_day)],
                lead_day_to_index,
            ).assign(first_day=pandas.Timestamp(first_day))
            for first_day, first_day_group in observations_dataframe.groupby("first_day", sort=True)
        ],
        ignore_index=True,
    )


def _class4_table_from_scores(
    scores: pandas.DataFrame,
    statistics: pandas.DataFrame,
    value_column: str,
    variable_key: str,
) -> pandas.DataFrame:
    counts = (
        statistics.loc[statistics["width_km"] == 0.0]
        .groupby(["depth_bin", "lead_day"], as_index=False)[["total", "count"]]
        .sum()
    )
    merged = scores.merge(counts, on=["depth_bin", "lead_day"])
    return pandas.DataFrame(
        {
            "variable": variable_key,
            "depth_bin": merged["depth_bin"],
            "lead_day": merged["lead_day"],
            "rmsd": merged[value_column],
            "count": merged["total"].astype(int),
            "missing": (merged["total"] - merged["count"]).astype(int),
        }
    )


def smoothing_controlled_rmsd_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    ocean_mask: xarray.DataArray,
    variables: list[Variable],
    climatology_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    """
    Smoothing-controlled RMSD table, shaped like the Class IV RMSD table.

    The fitted width and slope of each cell are in ``table.attrs["row_diagnostics"]``, two tables of the same shape.
    """
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    climatology = rename_dataset_with_standard_names(climatology_dataset)
    lead_days_count = challenger.sizes[Dimension.LEAD_DAY_INDEX.key()]
    population_layers = class4_population_layers(ocean_mask)
    tables = {"rmsd": [], "width_km": [], "slope": []}
    for variable in variables:
        variable_key = variable.key()
        observations_dataframe = create_class4_observations_dataframe(
            reference_dataset, variable_key, variable_key, lead_days_count
        )
        if observations_dataframe.empty:
            continue
        observations_dataframe = class4_observations_in_shared_population(
            observations_dataframe.dropna(subset=["observation_value"]),
            population_layers,
        )
        if observations_dataframe.empty:
            continue
        statistics = _variable_sufficient_statistics(
            prepare_class4_model_variable(challenger[variable_key], variable_key),
            prepare_climatology_variable(climatology[variable_key], variable_key),
            observations_dataframe,
        )
        scores = score_smoothing_controlled_rmsd(statistics)
        for value_column, table in tables.items():
            table.append(_class4_table_from_scores(scores, statistics, value_column, variable_key))
    if not tables["rmsd"]:
        return pandas.DataFrame()
    formatted = {
        value_column: format_class4_results(pandas.concat(table, ignore_index=True), lead_days_count)
        for value_column, table in tables.items()
    }
    result = formatted["rmsd"]
    result.attrs[ROW_DIAGNOSTICS_ATTRIBUTE] = {
        SMOOTHING_WIDTH_ROW_FIELD: formatted["width_km"].drop(columns=result.columns[-2:]),
        SMOOTHING_SLOPE_ROW_FIELD: formatted["slope"].drop(columns=result.columns[-2:]),
    }
    return result
