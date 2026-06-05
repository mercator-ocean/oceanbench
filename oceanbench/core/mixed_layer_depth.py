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


def compute_mixed_layer_depth(dataset: xarray.Dataset) -> xarray.Dataset:
    return _compute_mixed_layer_depth(_cap_depth(_harmonise_dataset(dataset)))


def _harmonise_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    return rename_dataset_with_standard_names(dataset)


def _cap_depth(dataset: xarray.Dataset) -> xarray.Dataset:
    depth = dataset[Dimension.DEPTH.key()]
    return dataset.where(depth <= MAXIMUM_MIXED_LAYER_DEPTH, drop=True)


def _compute_absolute_salinity(
    salinity: xarray.DataArray,
    depth: xarray.DataArray,
    longitude: xarray.DataArray,
    latitude: xarray.DataArray,
) -> xarray.DataArray:
    return gsw.SA_from_SP(salinity, depth, longitude, latitude)


def _compute_potential_density(
    absolute_salinity: xarray.DataArray,
    temperature: xarray.DataArray,
    depth: xarray.DataArray,
) -> xarray.DataArray:
    absolute_salinity = absolute_salinity.clip(min=0)  # filter out negative salinities

    return gsw.pot_rho_t_exact(absolute_salinity, temperature, depth, 0)


def _compute_mixed_layer_depth(dataset: xarray.Dataset) -> xarray.Dataset:
    density_threshold = 0.03  # kg/m^3 threshold for MLD definition
    temperature = dataset[Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()]
    salinity = dataset[Variable.SEA_WATER_SALINITY.key()]
    depth = dataset[Dimension.DEPTH.key()]
    latitude = dataset[Dimension.LATITUDE.key()]
    longitude = dataset[Dimension.LONGITUDE.key()]
    absolute_salinity = _compute_absolute_salinity(salinity, depth, longitude, latitude)
    potential_density = _compute_potential_density(absolute_salinity, temperature, depth)
    surface_density = potential_density.isel({Dimension.DEPTH.key(): 0})

    delta_density = potential_density - surface_density
    mask = delta_density >= density_threshold
    threshold_crossed = mask.any(dim=Dimension.DEPTH.key())

    threshold_mixed_layer_depth_index = mask.argmax(dim=Dimension.DEPTH.key())
    deepest_valid_depth_index = _deepest_valid_depth_index(temperature)
    mixed_layer_depth_index = threshold_mixed_layer_depth_index.where(threshold_crossed, deepest_valid_depth_index)
    mixed_layer_depth_depth = _depths_for_indices(depth, mixed_layer_depth_index).assign_attrs(
        {"standard_name": StandardVariable.MIXED_LAYER_THICKNESS.value}
    )
    temperature_mask = xarray.ufuncs.isfinite(temperature.isel({Dimension.DEPTH.key(): 0}))

    masked_mixed_layer_depth = mixed_layer_depth_depth.where(temperature_mask)

    return xarray.Dataset(
        data_vars={Variable.MIXED_LAYER_DEPTH.key(): masked_mixed_layer_depth},
        coords=dataset.coords,
    )


def _deepest_valid_depth_index(temperature: xarray.DataArray) -> xarray.DataArray:
    depth_dimension = Dimension.DEPTH.key()
    reversed_valid_temperature = xarray.ufuncs.isfinite(temperature).isel({depth_dimension: slice(None, None, -1)})
    reversed_index = reversed_valid_temperature.argmax(dim=depth_dimension)
    return temperature.sizes[depth_dimension] - 1 - reversed_index


def _depths_for_indices(depth: xarray.DataArray, indices: xarray.DataArray) -> xarray.DataArray:
    dask_depth = xarray.DataArray(dask.array.asarray(depth.data), dims=depth.dims)
    return dask_depth.isel({Dimension.DEPTH.key(): indices})
