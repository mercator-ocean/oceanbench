# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import xarray

from oceanbench.core import mixed_layer_depth
from oceanbench.core import geostrophic_currents


def add_mixed_layer_depth(
    dataset: xarray.Dataset,
) -> xarray.Dataset:
    return mixed_layer_depth.add_mixed_layer_depth(dataset)


def add_geostrophic_currents(
    dataset: xarray.Dataset,
) -> xarray.Dataset:
    return geostrophic_currents.add_geostrophic_currents(dataset)
