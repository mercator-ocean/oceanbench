# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import gsw
import xarray

from oceanbench.core.climate_forecast_standard_names import (
    StandardVariable,
    remane_dataset_with_standard_names,
)
from oceanbench.core.dataset_utils import (
    Dimension,
    Variable,
)


def compute_mixed_layer_depth(dataset: xarray.Dataset) -> xarray.Dataset:
    return _compute_mixed_layer_depth(_harmonise_dataset(dataset))


def _harmonise_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    return remane_dataset_with_standard_names(dataset)


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
    return gsw.pot_rho_t_exact(absolute_salinity, temperature, depth, 0)


def _compute_mixed_layer_depth(dataset: xarray.Dataset) -> xarray.Dataset:
    print(f"{dataset=}")
    density_threshold = 0.03  # kg/m^3 threshold for MLD definition
    temperature = dataset[Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()]
    print(f"{type(temperature.data)=}")
    salinity = dataset[Variable.SEA_WATER_SALINITY.key()]
    depth = dataset[Dimension.DEPTH.key()]
    latitude = dataset[Dimension.LATITUDE.key()]
    longitude = dataset[Dimension.LONGITUDE.key()]
    absolute_salinity = _compute_absolute_salinity(salinity, depth, longitude, latitude)
    potential_density = _compute_potential_density(absolute_salinity, temperature, depth)
    surface_density = potential_density.isel({Dimension.DEPTH.key(): 0})
    delta_density = potential_density - surface_density
    mask = delta_density >= density_threshold
    mixed_layer_depth_index = mask.argmax(dim=Dimension.DEPTH.key())
    print("pre-map_blocks")

    def select_depth(index):
        return depth.isel({Dimension.DEPTH.key(): index})

    print(f"{type(depth.data)=}")
    mixed_layer_depth_depth = xarray.DataArray(
        depth.values[mixed_layer_depth_index.values],
        coords=mixed_layer_depth_index.coords,
        dims=mixed_layer_depth_index.dims,
        attrs={"standard_name": StandardVariable.MIXED_LAYER_THICKNESS.value}
    )
    print("post-map_blocks")
    temperature_mask = xarray.ufuncs.isfinite(temperature.isel({Dimension.DEPTH.key(): 0}))

    print("pre-mask")
    masked_mixed_layer_depth = mixed_layer_depth_depth.where(temperature_mask)

    dataset_res = xarray.Dataset(
        data_vars={Variable.MIXED_LAYER_DEPTH.key(): masked_mixed_layer_depth},
        coords=dataset.coords,
    )
    print(dataset_res)
    return dataset_res
