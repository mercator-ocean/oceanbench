# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import gsw
import xarray
import dask

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
DENSITY_THRESHOLD = 0.03


def compute_mixed_layer_depth(dataset: xarray.Dataset) -> xarray.Dataset:
    return _compute_mixed_layer_depth(_cap_depth(_harmonise_dataset(dataset)))


def _harmonise_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    return rename_dataset_with_standard_names(dataset)


def _cap_depth(dataset: xarray.Dataset) -> xarray.Dataset:
    depth_dimension = Dimension.DEPTH.key()
    depth = dataset[depth_dimension]
    return dataset.isel({depth_dimension: (depth <= MAXIMUM_MIXED_LAYER_DEPTH).values})


def _compute_potential_density_anomaly(
    practical_salinity: xarray.DataArray,
    potential_temperature: xarray.DataArray,
    depth: xarray.DataArray,
    longitude: xarray.DataArray,
    latitude: xarray.DataArray,
) -> xarray.DataArray:
    pressure = gsw.p_from_z(-depth, latitude)
    absolute_salinity = gsw.SA_from_SP(practical_salinity, pressure, longitude, latitude).clip(min=0)
    conservative_temperature = gsw.CT_from_pt(absolute_salinity, potential_temperature)
    return gsw.sigma0(absolute_salinity, conservative_temperature)


def _compute_mixed_layer_depth(dataset: xarray.Dataset) -> xarray.Dataset:
    temperature = dataset[Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()]
    depth = dataset[Dimension.DEPTH.key()]
    potential_density_anomaly = _compute_potential_density_anomaly(
        dataset[Variable.SEA_WATER_SALINITY.key()],
        temperature,
        depth,
        dataset[Dimension.LONGITUDE.key()],
        dataset[Dimension.LATITUDE.key()],
    )
    threshold_mixed_layer_depth = _threshold_crossing_depth(potential_density_anomaly, depth)
    deepest_valid_depth = _depths_for_indices(depth, _deepest_valid_depth_index(temperature))
    unmasked_mixed_layer_depth = threshold_mixed_layer_depth.fillna(deepest_valid_depth).assign_attrs(
        {"standard_name": StandardVariable.MIXED_LAYER_THICKNESS.value}
    )
    temperature_mask = xarray.ufuncs.isfinite(temperature.isel({Dimension.DEPTH.key(): 0}))

    masked_mixed_layer_depth = unmasked_mixed_layer_depth.where(temperature_mask)

    return xarray.Dataset(
        data_vars={Variable.MIXED_LAYER_DEPTH.key(): masked_mixed_layer_depth},
        coords=dataset.coords,
    )


def _threshold_crossing_depth(
    potential_density_anomaly: xarray.DataArray, native_depth: xarray.DataArray
) -> xarray.DataArray:
    depth_dimension = Dimension.DEPTH.key()
    depth = native_depth.astype("float64")
    reference_density = (
        potential_density_anomaly.assign_coords({depth_dimension: depth})
        .interp({depth_dimension: REFERENCE_DEPTH})
        .drop_vars(depth_dimension)
    )
    delta_density = potential_density_anomaly - reference_density
    level_is_below_reference = depth > REFERENCE_DEPTH
    shallower_depth = depth.shift({depth_dimension: 1})
    shallower_level_is_below_reference = shallower_depth > REFERENCE_DEPTH
    segment_top_depth = shallower_depth.where(shallower_level_is_below_reference, REFERENCE_DEPTH)
    segment_top_delta_density = delta_density.shift({depth_dimension: 1}).where(shallower_level_is_below_reference, 0)
    crosses_threshold = (
        level_is_below_reference
        & (segment_top_delta_density < DENSITY_THRESHOLD)
        & (delta_density >= DENSITY_THRESHOLD)
    )
    crossing_segment_delta_density = (delta_density - segment_top_delta_density).where(crosses_threshold)
    crossing_depth = (
        segment_top_depth
        + (DENSITY_THRESHOLD - segment_top_delta_density) * (depth - segment_top_depth) / crossing_segment_delta_density
    )
    return crossing_depth.min(dim=depth_dimension)


def _deepest_valid_depth_index(temperature: xarray.DataArray) -> xarray.DataArray:
    depth_dimension = Dimension.DEPTH.key()
    reversed_valid_temperature = xarray.ufuncs.isfinite(temperature).isel({depth_dimension: slice(None, None, -1)})
    reversed_index = reversed_valid_temperature.argmax(dim=depth_dimension)
    return temperature.sizes[depth_dimension] - 1 - reversed_index


def _depths_for_indices(depth: xarray.DataArray, indices: xarray.DataArray) -> xarray.DataArray:
    dask_depth = xarray.DataArray(dask.array.asarray(depth.data), dims=depth.dims)
    return dask_depth.isel({Dimension.DEPTH.key(): indices})
