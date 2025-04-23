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


def add_mixed_layer_depth(dataset: xarray.Dataset) -> xarray.Dataset:
    density_threshold = 0.03  # kg/m^3 threshold for MLD definition
    temperature = get_variable(dataset, Variable.TEMPERATURE)
    salinity = get_variable(dataset, Variable.SALINITY)
    depth = get_dimension(dataset, Dimension.DEPTH)
    latitude = get_dimension(dataset, Dimension.LATITUDE)
    longitude = get_dimension(dataset, Dimension.LONGITUDE)
    absolute_salinity = gsw.SA_from_SP(salinity, depth, longitude, latitude)
    potential_density = gsw.pot_rho_t_exact(absolute_salinity, temperature, depth, 0)
    surface_density = potential_density.isel(depth=0)
    delta_density = potential_density - surface_density
    mask = delta_density >= density_threshold
    mixed_layer_depth_index = mask.argmax(dim=Dimension.DEPTH.dimension_name_from_dataset(dataset))
    mixed_layer_depth_depth = depth.isel(depth=mixed_layer_depth_index)
    temperature_mask = numpy.isfinite(temperature.isel(depth=0))

    return dataset.assign({Variable.MIXED_LAYER_DEPTH.value: mixed_layer_depth_depth.where(temperature_mask)})
