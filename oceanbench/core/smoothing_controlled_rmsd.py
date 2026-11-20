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
import scipy.fft
import xarray

from oceanbench.core.classIV_support import (
    _bracketing_level_indices,
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
from oceanbench.core.dataset_utils import Dimension, Variable, is_global_longitude_grid
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

# "grid" smooths each field once on its grid and samples it like the point value; "stencil" is the reference path
# that samples the field at the stencil points around every observation.
GRID_SMOOTHING_METHOD = "grid"
STENCIL_SMOOTHING_METHOD = "stencil"
SMOOTHING_METHODS = (GRID_SMOOTHING_METHOD, STENCIL_SMOOTHING_METHOD)
# Grid cells whose smoothed ocean fraction is below this have no Gaussian mean (all land within a few sigma).
_MINIMUM_OCEAN_FRACTION = 1e-4
# Cosine of latitude floor, so the zonal width in cells stays finite at the poles.
_MINIMUM_LATITUDE_COSINE = 1e-3
# Gaussian tails kept by the zero padding of non periodic axes, in sigmas.
_PADDING_SIGMAS = 4.0
# Climatology margin around the observations, in degrees, when it is cut before smoothing.
_CLIMATOLOGY_MARGIN_DEGREES = 6.0
_LEVEL_CHUNK_SIZE = 8


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


def _gaussian_transfer(frequencies: numpy.ndarray, sigma_cells: numpy.ndarray) -> numpy.ndarray:
    return numpy.exp(-2 * numpy.pi**2 * frequencies**2 * sigma_cells**2)


def grid_gaussian_means(
    values: numpy.ndarray,
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
) -> numpy.ndarray:
    """
    Land aware normalised Gaussian means of a field on its latitude longitude grid, one per SIGMAS_KILOMETERS.

    ``values`` has the latitude and longitude as its last two axes; non finite values are land. The Gaussian has the
    same standard deviation in km everywhere: a zonal pass with a width in cells that depends on the latitude, then a
    meridional pass, both by the exact Gaussian transfer function in Fourier space. Longitude is periodic on a global
    grid only; other axes are zero padded, and the ocean weights are smoothed the same way to renormalise.
    The result has a leading axis of SIGMAS_KILOMETERS, NaN where no ocean lies within reach.
    """
    latitude_count, longitude_count = values.shape[-2:]
    latitude_step_kilometers = abs(float(latitudes[1] - latitudes[0])) * KILOMETERS_PER_DEGREE
    longitude_step_degrees = abs(float(longitudes[1] - longitudes[0]))
    row_cosines = numpy.maximum(numpy.cos(numpy.deg2rad(latitudes)), _MINIMUM_LATITUDE_COSINE)
    largest_sigma = max(SIGMAS_KILOMETERS)
    latitude_padding = min(int(numpy.ceil(_PADDING_SIGMAS * largest_sigma / latitude_step_kilometers)), latitude_count)
    if is_global_longitude_grid(numpy.asarray(longitudes)):
        longitude_padding = 0
    else:
        smallest_step = longitude_step_degrees * KILOMETERS_PER_DEGREE * row_cosines.min()
        longitude_padding = min(int(numpy.ceil(_PADDING_SIGMAS * largest_sigma / smallest_step)), longitude_count)
    is_ocean = numpy.isfinite(values)
    stacked = numpy.stack([numpy.where(is_ocean, values, 0.0), is_ocean.astype(float)])
    padding = [(0, 0)] * (stacked.ndim - 2) + [(latitude_padding,) * 2, (longitude_padding,) * 2]
    stacked = numpy.pad(stacked, padding)
    padded_latitudes, padded_longitudes = stacked.shape[-2:]
    row_steps_kilometers = numpy.pad(
        longitude_step_degrees * KILOMETERS_PER_DEGREE * row_cosines, latitude_padding, mode="edge"
    )
    zonal_spectrum = scipy.fft.rfft(stacked, axis=-1, workers=-1)
    zonal_frequencies = numpy.fft.rfftfreq(padded_longitudes)
    meridional_frequencies = numpy.fft.rfftfreq(padded_latitudes)
    crop = (
        Ellipsis,
        slice(latitude_padding, latitude_padding + latitude_count),
        slice(longitude_padding, longitude_padding + longitude_count),
    )
    means = numpy.empty((len(SIGMAS_KILOMETERS),) + values.shape)
    for index, sigma in enumerate(SIGMAS_KILOMETERS):
        zonal = scipy.fft.irfft(
            zonal_spectrum * _gaussian_transfer(zonal_frequencies[None, :], sigma / row_steps_kilometers[:, None]),
            n=padded_longitudes,
            axis=-1,
            workers=-1,
        )
        smoothed = scipy.fft.irfft(
            scipy.fft.rfft(zonal, axis=-2, workers=-1)
            * _gaussian_transfer(meridional_frequencies, sigma / latitude_step_kilometers)[:, None],
            n=padded_latitudes,
            axis=-2,
            workers=-1,
        )[crop]
        weights = smoothed[1]
        means[index] = numpy.where(
            weights > _MINIMUM_OCEAN_FRACTION,
            smoothed[0] / numpy.maximum(weights, _MINIMUM_OCEAN_FRACTION),
            numpy.nan,
        )
    return means


def _levels_bracketing(field: xarray.DataArray, observation_depths: numpy.ndarray | None) -> xarray.DataArray:
    """Keep the model levels that bracket some observation depth, the only ones the vertical interpolation reads."""
    depth_key = Dimension.DEPTH.key()
    if observation_depths is None or depth_key not in field.dims or field.sizes[depth_key] == 1:
        return field
    model_depths = field[depth_key].values
    sort_order = numpy.argsort(model_depths)
    lower, upper = _bracketing_level_indices(model_depths[sort_order], numpy.unique(observation_depths))
    return field.isel({depth_key: numpy.sort(sort_order[numpy.unique(numpy.concatenate([lower, upper]))])})


def grid_smoothed_fields(
    field: xarray.DataArray,
    observation_depths: numpy.ndarray | None = None,
) -> list[xarray.DataArray]:
    """
    The Gaussian means of ``field`` on its grid, one field per SIGMAS_KILOMETERS, with the coordinates of ``field``.

    With ``observation_depths``, only the levels bracketing those depths are smoothed and kept.
    """
    field = _levels_bracketing(field, observation_depths)
    latitude_key, longitude_key = Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()
    other_dimensions = [dimension for dimension in field.dims if dimension not in (latitude_key, longitude_key)]
    field = field.transpose(*other_dimensions, latitude_key, longitude_key)
    values = numpy.asarray(field.values, dtype=float)
    latitudes, longitudes = field[latitude_key].values, field[longitude_key].values
    if other_dimensions:
        flat = values.reshape(-1, *values.shape[-2:])
        means = numpy.concatenate(
            [
                grid_gaussian_means(flat[start : start + _LEVEL_CHUNK_SIZE], latitudes, longitudes)
                for start in range(0, flat.shape[0], _LEVEL_CHUNK_SIZE)
            ],
            axis=1,
        ).reshape((len(SIGMAS_KILOMETERS),) + values.shape)
    else:
        means = grid_gaussian_means(values, latitudes, longitudes)
    return [field.copy(data=sigma_means) for sigma_means in means]


def _gaussian_means_from_smoothed_fields(
    smoothed_fields: list[xarray.DataArray],
    observation_group: pandas.DataFrame,
) -> numpy.ndarray:
    return numpy.stack(
        [
            _interpolated_model_values_for_observation_group(
                smoothed_field, observation_group, smoothed_field[Dimension.DEPTH.key()].values
            )
            for smoothed_field in smoothed_fields
        ]
    )


def gaussian_means(
    field: xarray.DataArray,
    observation_group: pandas.DataFrame,
    model_depths: numpy.ndarray,
    smoothing_method: str,
    smoothed_fields: list[xarray.DataArray] | None = None,
) -> numpy.ndarray:
    """Gaussian means at the observations, one row per sigma, by either method; grid fields may be passed in."""
    if smoothing_method == STENCIL_SMOOTHING_METHOD:
        return gaussian_means_at_observations(field, observation_group, model_depths)
    if smoothed_fields is None:
        smoothed_fields = grid_smoothed_fields(field, observation_group[Dimension.DEPTH.key()].values)
    return _gaussian_means_from_smoothed_fields(smoothed_fields, observation_group)


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


def _climatology_near_observations(
    climatology_variable: xarray.DataArray,
    observations_dataframe: pandas.DataFrame,
) -> xarray.DataArray:
    """Cut the climatology to the observations plus a margin, so a regional evaluation smooths a regional field."""
    latitude_key, longitude_key = Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()
    latitudes = observations_dataframe[latitude_key].values
    longitudes = observations_dataframe[longitude_key].values
    grid_latitudes = climatology_variable[latitude_key].values
    grid_longitudes = climatology_variable[longitude_key].values
    is_kept_latitude = (grid_latitudes >= latitudes.min() - _CLIMATOLOGY_MARGIN_DEGREES) & (
        grid_latitudes <= latitudes.max() + _CLIMATOLOGY_MARGIN_DEGREES
    )
    climatology_variable = climatology_variable.isel({latitude_key: numpy.flatnonzero(is_kept_latitude)})
    first_longitude = longitudes.min() - _CLIMATOLOGY_MARGIN_DEGREES
    last_longitude = longitudes.max() + _CLIMATOLOGY_MARGIN_DEGREES
    if first_longitude > grid_longitudes.min() and last_longitude < grid_longitudes.max():
        is_kept_longitude = (grid_longitudes >= first_longitude) & (grid_longitudes <= last_longitude)
        climatology_variable = climatology_variable.isel({longitude_key: numpy.flatnonzero(is_kept_longitude)})
    return climatology_variable


def climatology_at_observations(
    climatology_variable: xarray.DataArray,
    observations_dataframe: pandas.DataFrame,
    smoothing_method: str = GRID_SMOOTHING_METHOD,
    month_cache: dict | None = None,
) -> numpy.ndarray:
    """
    Monthly climatology at each observation, linear in time between mid months, one row per smoothing width.

    The first row is the Class IV point value, the next rows are the Gaussian means of SIGMAS_KILOMETERS with the
    same method as the forecast, so the smoothing applies to the forecast departure from climatology.
    The climatology has one time step per calendar month. ``month_cache`` keeps each month's fields (and its grid
    Gaussian means) across calls on the same climatology variable.
    """
    time_key = Dimension.TIME.key()
    month_cache = {} if month_cache is None else month_cache
    climatology_variable = _model_data_with_depth_dimension(climatology_variable)
    month_order = numpy.argsort(pandas.DatetimeIndex(climatology_variable[time_key].values).month.values)
    climatology_variable = climatology_variable.isel({time_key: month_order})
    climatology_depths = climatology_variable[Dimension.DEPTH.key()].values
    lower_months, upper_months, upper_weights = _monthly_interpolation(observations_dataframe[time_key].values)
    lower_values = numpy.full((len(SMOOTHING_WIDTHS_KILOMETERS), len(observations_dataframe)), numpy.nan)
    upper_values = numpy.full_like(lower_values, numpy.nan)
    for month in numpy.unique(numpy.concatenate([lower_months, upper_months])):
        if month not in month_cache:
            month_slice = climatology_variable.isel({time_key: int(month)}).compute()
            month_cache[month] = (
                month_slice,
                grid_smoothed_fields(month_slice) if smoothing_method == GRID_SMOOTHING_METHOD else None,
            )
        month_slice, smoothed_fields = month_cache[month]
        needed_indices = numpy.flatnonzero((lower_months == month) | (upper_months == month))
        month_observations = observations_dataframe.iloc[needed_indices]
        month_values = numpy.vstack(
            [
                _interpolated_model_values_for_observation_group(month_slice, month_observations, climatology_depths)[
                    None, :
                ],
                gaussian_means(month_slice, month_observations, climatology_depths, smoothing_method, smoothed_fields),
            ]
        )
        is_lower = lower_months[needed_indices] == month
        is_upper = upper_months[needed_indices] == month
        lower_values[:, needed_indices[is_lower]] = month_values[:, is_lower]
        upper_values[:, needed_indices[is_upper]] = month_values[:, is_upper]
    return lower_values + upper_weights[None, :] * (upper_values - lower_values)


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
    """
    ``climatology_values`` has one row per smoothing width, the point value first, like the forecast rows.

    The forecast anomaly at width s is G_s f - G_s c. Where either Gaussian mean is missing (no ocean sample),
    the point anomaly is used.
    """
    forecasts = numpy.vstack([point_values[None, :], gaussian_means])
    point_climatology = climatology_values[0]
    is_scored = numpy.isfinite(point_values) & numpy.isfinite(point_climatology)
    observation_anomalies = (observations - point_climatology)[is_scored]
    point_anomalies = (point_values - point_climatology)[is_scored]
    smoothed_anomalies = forecasts[:, is_scored] - climatology_values[:, is_scored]
    forecast_anomalies = numpy.where(numpy.isfinite(smoothed_anomalies), smoothed_anomalies, point_anomalies)
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
    smoothing_method: str = GRID_SMOOTHING_METHOD,
) -> pandas.DataFrame:
    """
    Per depth bin, lead day and smoothing width sufficient statistics of one forecast initialization day.

    ``climatology_values`` has one row per smoothing width and one column per row of ``first_day_group``.
    Observations whose point value or climatology
    is missing are not counted, as the plain RMSD leaves out observations without a point value.
    """
    model_depths = model_data[Dimension.DEPTH.key()].values
    first_day_block = model_data.isel({Dimension.FIRST_DAY_DATETIME.key(): first_day_index}).compute()
    records = []
    group_positions = pandas.Series(numpy.arange(len(first_day_group)), index=first_day_group.index)
    smoothed_by_lead_day: dict[object, list[xarray.DataArray]] = {}
    for (lead_day, depth_bin), observation_group in first_day_group.groupby(["lead_day", "depth_bin"], sort=True):
        time_slice = first_day_block.isel({Dimension.LEAD_DAY_INDEX.key(): lead_day_to_index[lead_day]})
        if smoothing_method == GRID_SMOOTHING_METHOD and lead_day not in smoothed_by_lead_day:
            smoothed_by_lead_day.clear()
            smoothed_by_lead_day[lead_day] = grid_smoothed_fields(
                time_slice, first_day_group[Dimension.DEPTH.key()].values
            )
        statistics = _sufficient_statistics_of_group(
            observation_group["observation_value"].values,
            _interpolated_model_values_for_observation_group(time_slice, observation_group, model_depths),
            gaussian_means(
                time_slice, observation_group, model_depths, smoothing_method, smoothed_by_lead_day.get(lead_day)
            ),
            climatology_values[:, group_positions.loc[observation_group.index].values],
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
    smoothing_method: str = GRID_SMOOTHING_METHOD,
) -> pandas.DataFrame:
    observations_dataframe = observations_dataframe.reset_index(drop=True)
    climatology_variable = _levels_bracketing(
        _model_data_with_depth_dimension(_climatology_near_observations(climatology_variable, observations_dataframe)),
        observations_dataframe[Dimension.DEPTH.key()].values,
    )
    month_cache: dict = {}
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
                climatology_at_observations(climatology_variable, first_day_group, smoothing_method, month_cache),
                first_day_group,
                first_day_to_index[pandas.Timestamp(first_day)],
                lead_day_to_index,
                smoothing_method,
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
    smoothing_method: str = GRID_SMOOTHING_METHOD,
) -> pandas.DataFrame:
    """
    Smoothing-controlled RMSD table, shaped like the Class IV RMSD table.

    ``smoothing_method`` is "grid" (default, each field smoothed once on its grid) or "stencil" (reference path).

    The fitted width and slope of each cell are in ``table.attrs["row_diagnostics"]``, two tables of the same shape.
    """
    if smoothing_method not in SMOOTHING_METHODS:
        raise ValueError(f"smoothing_method must be one of {SMOOTHING_METHODS}, not {smoothing_method!r}.")
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
            smoothing_method,
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
