# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import gsw
import xarray

from oceanbench.core.dataset_utils import (
    Dimension,
    Variable,
    get_dimension,
    get_variable,
)


def compute_mixed_layer_depth(dataset: xarray.Dataset) -> xarray.Dataset:
    return _compute_mixed_layer_depth(dataset)


def compute_absolute_salinity(
    salinity: xarray.DataArray,
    depth: xarray.DataArray,
    longitude: xarray.DataArray,
    latitude: xarray.DataArray,
) -> xarray.DataArray:
    return gsw.SA_from_SP(salinity, depth, longitude, latitude)


def compute_potential_density(
    absolute_salinity: xarray.DataArray,
    temperature: xarray.DataArray,
    depth: xarray.DataArray,
) -> xarray.DataArray:
    return gsw.pot_rho_t_exact(absolute_salinity, temperature, depth, 0).compute()


def _compute_mixed_layer_depth(dataset: xarray.Dataset) -> xarray.Dataset:
    density_threshold = 0.03  # kg/m^3 threshold for MLD definition
    temperature = get_variable(dataset, Variable.TEMPERATURE)
    salinity = get_variable(dataset, Variable.SALINITY)
    depth = get_dimension(dataset, Dimension.DEPTH)
    latitude = get_dimension(dataset, Dimension.LATITUDE)
    longitude = get_dimension(dataset, Dimension.LONGITUDE)
    absolute_salinity = compute_absolute_salinity(salinity, depth, longitude, latitude)
    potential_density = compute_potential_density(absolute_salinity, temperature, depth)
    surface_density = potential_density.isel({Dimension.DEPTH.key(): 0})
    delta_density = potential_density - surface_density
    mask = delta_density >= density_threshold
    mixed_layer_depth_index = mask.argmax(dim=Dimension.DEPTH.dimension_name_from_dataset(dataset))
    mixed_layer_depth_depth = depth.isel({Dimension.DEPTH.key(): mixed_layer_depth_index}).assign_attrs(
        {"standard_name": "ocean_mixed_layer_thickness"}
    )
    temperature_mask = numpy.isfinite(temperature.isel({Dimension.DEPTH.key(): 0}))

    return xarray.Dataset(
        data_vars={Variable.MIXED_LAYER_DEPTH.key(): mixed_layer_depth_depth.where(temperature_mask)},
        coords=dataset.coords,
    )
