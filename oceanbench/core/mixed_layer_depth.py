# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import gsw
import numpy
import xarray

from oceanbench.core.climate_forecast_standard_names import (
    StandardVariable,
    rename_dataset_with_standard_names,
)
from oceanbench.core.dataset_utils import (
    Dimension,
    Variable,
)

MAXIMUM_MIXED_LAYER_DEPTH = 600.0
REFERENCE_DEPTH = 10.0
DENSITY_THRESHOLD = 0.03  # kg/m^3, de Boyer Montegut et al. (2004)


def compute_mixed_layer_depth(dataset: xarray.Dataset) -> xarray.Dataset:
    return _compute_mixed_layer_depth(keep_mixed_layer_depth_levels(_harmonise_dataset(dataset)))


def keep_mixed_layer_depth_levels(dataset: xarray.Dataset) -> xarray.Dataset:
    """Keep the levels down to the first level at or below the mixed layer depth cap."""
    depth_dimension = Dimension.DEPTH.key()
    depths = dataset[depth_dimension].values
    levels_count = min(int(numpy.searchsorted(depths, MAXIMUM_MIXED_LAYER_DEPTH, side="left")) + 1, depths.size)
    return dataset.isel({depth_dimension: slice(0, levels_count)})


def _harmonise_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    return rename_dataset_with_standard_names(dataset)


def _compute_absolute_salinity(
    salinity: xarray.DataArray,
    pressure: xarray.DataArray,
    longitude: xarray.DataArray,
    latitude: xarray.DataArray,
) -> xarray.DataArray:
    return gsw.SA_from_SP(salinity, pressure, longitude, latitude)


def _compute_potential_density_anomaly(
    salinity: xarray.DataArray,
    potential_temperature: xarray.DataArray,
    depth: xarray.DataArray,
    longitude: xarray.DataArray,
    latitude: xarray.DataArray,
) -> xarray.DataArray:
    pressure = gsw.p_from_z(-depth, latitude)
    absolute_salinity = _compute_absolute_salinity(salinity, pressure, longitude, latitude)
    absolute_salinity = absolute_salinity.clip(min=0)  # filter out negative salinities
    conservative_temperature = gsw.CT_from_pt(absolute_salinity, potential_temperature)
    return gsw.sigma0(absolute_salinity, conservative_temperature)


def _mixed_layer_depth_of_profiles(sigma0: numpy.ndarray, depths: numpy.ndarray) -> numpy.ndarray:
    """Threshold mixed layer depth along the last axis of sigma0, levels ordered from the surface down."""
    valid = numpy.isfinite(sigma0)
    any_valid = valid.any(axis=-1)
    deepest_valid_index = depths.size - 1 - numpy.argmax(valid[..., ::-1], axis=-1)
    deepest_valid_depth = numpy.where(any_valid, depths[deepest_valid_index], numpy.nan)

    first_level_at_or_below_reference = int(numpy.searchsorted(depths, REFERENCE_DEPTH, side="left"))
    if first_level_at_or_below_reference == depths.size:
        return numpy.minimum(deepest_valid_depth, MAXIMUM_MIXED_LAYER_DEPTH)
    lower_index = first_level_at_or_below_reference
    upper_index = max(lower_index - 1, 0)
    lower_depth, upper_depth = depths[lower_index], depths[upper_index]
    weight = 0.0 if lower_depth == upper_depth else (REFERENCE_DEPTH - upper_depth) / (lower_depth - upper_depth)
    reference_sigma0 = sigma0[..., upper_index] + weight * (sigma0[..., lower_index] - sigma0[..., upper_index])

    below_reference = depths > REFERENCE_DEPTH
    profile_depths = numpy.concatenate([[REFERENCE_DEPTH], depths[below_reference]])
    profile_sigma0 = numpy.concatenate([reference_sigma0[..., None], sigma0[..., below_reference]], axis=-1)
    threshold = reference_sigma0[..., None] + DENSITY_THRESHOLD
    crossed = profile_sigma0 >= threshold
    threshold_crossed = crossed.any(axis=-1)
    crossing_index = numpy.maximum(numpy.argmax(crossed, axis=-1), 1)

    deeper_depth = profile_depths[crossing_index]
    shallower_depth = profile_depths[crossing_index - 1]
    deeper_sigma0 = numpy.take_along_axis(profile_sigma0, crossing_index[..., None], axis=-1)[..., 0]
    shallower_sigma0 = numpy.take_along_axis(profile_sigma0, crossing_index[..., None] - 1, axis=-1)[..., 0]
    with numpy.errstate(invalid="ignore", divide="ignore"):
        crossing_fraction = (threshold[..., 0] - shallower_sigma0) / (deeper_sigma0 - shallower_sigma0)
    crossing_depth = shallower_depth + crossing_fraction * (deeper_depth - shallower_depth)

    mixed_layer_depth = numpy.where(
        threshold_crossed & (deepest_valid_depth >= REFERENCE_DEPTH), crossing_depth, deepest_valid_depth
    )
    return numpy.minimum(mixed_layer_depth, MAXIMUM_MIXED_LAYER_DEPTH)


def _compute_mixed_layer_depth(dataset: xarray.Dataset) -> xarray.Dataset:
    depth_dimension = Dimension.DEPTH.key()
    temperature = dataset[Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()]
    depth = dataset[depth_dimension]
    sigma0 = _compute_potential_density_anomaly(
        dataset[Variable.SEA_WATER_SALINITY.key()],
        temperature,
        depth,
        dataset[Dimension.LONGITUDE.key()],
        dataset[Dimension.LATITUDE.key()],
    )

    mixed_layer_depth = xarray.apply_ufunc(
        _mixed_layer_depth_of_profiles,
        sigma0,
        kwargs={"depths": depth.values},
        input_core_dims=[[depth_dimension]],
        dask="parallelized",
        output_dtypes=[numpy.float64],
        dask_gufunc_kwargs={"allow_rechunk": True},
    ).assign_attrs({"standard_name": StandardVariable.MIXED_LAYER_THICKNESS.value})

    surface_mask = xarray.ufuncs.isfinite(temperature.isel({depth_dimension: 0}))

    return xarray.Dataset(
        data_vars={Variable.MIXED_LAYER_DEPTH.key(): mixed_layer_depth.where(surface_mask)},
        coords=dataset.coords,
    )
