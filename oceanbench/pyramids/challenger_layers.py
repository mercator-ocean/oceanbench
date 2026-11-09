# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import xarray

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import VARIABLE_METADATA, Dimension, Variable
from oceanbench.pyramids.builder import LEAD_DAY_DIMENSION, START_DATE_DIMENSION, VariableSpec

# The viewer maps show these variables at the surface (shallowest level) and the currents at 15 m.
_SURFACE_DEPTH_LABEL = "surface"
_FIFTEEN_METRE_DEPTH_LABEL = "15m"
_FIFTEEN_METRES = 15.0

_SURFACE_VARIABLES = [
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
    Variable.SEA_WATER_SALINITY.key(),
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
]
_FIFTEEN_METRE_VARIABLES = [
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
]

_DISPLAY_COLORMAP = {
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): "balance",
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): "thermal",
    Variable.SEA_WATER_SALINITY.key(): "haline",
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(): "balance",
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): "balance",
}


def _rename_to_viewer_coordinates(field: xarray.DataArray) -> xarray.DataArray:
    lead_days = field[Dimension.LEAD_DAY_INDEX.key()].values + 1
    return (
        field.rename(
            {
                Dimension.FIRST_DAY_DATETIME.key(): START_DATE_DIMENSION,
                Dimension.LEAD_DAY_INDEX.key(): LEAD_DAY_DIMENSION,
            }
        )
        .assign_coords({LEAD_DAY_DIMENSION: lead_days})
        .reset_coords(drop=True)
    )


def _surface_layer(dataset: xarray.Dataset, variable_key: str) -> xarray.DataArray:
    field = dataset[variable_key]
    if Dimension.DEPTH.key() in field.dims:
        field = field.isel({Dimension.DEPTH.key(): 0})
    return _rename_to_viewer_coordinates(field)


def _fifteen_metre_layer(dataset: xarray.Dataset, variable_key: str) -> xarray.DataArray:
    field = dataset[variable_key].interp({Dimension.DEPTH.key(): _FIFTEEN_METRES})
    return _rename_to_viewer_coordinates(field)


def viewer_layers(dataset: xarray.Dataset) -> tuple[xarray.Dataset, dict[str, VariableSpec]]:
    """Layer dataset and per-layer specs ready for :func:`builder.build_pyramid`."""
    standardised = rename_dataset_with_standard_names(dataset)
    layers: dict[str, xarray.DataArray] = {}
    specs: dict[str, VariableSpec] = {}

    for variable_key in _SURFACE_VARIABLES:
        if variable_key not in standardised.data_vars:
            continue
        layers[variable_key] = _surface_layer(standardised, variable_key)
        specs[variable_key] = VariableSpec(
            standard_name=variable_key,
            depth=_SURFACE_DEPTH_LABEL,
            units=VARIABLE_METADATA[variable_key][1],
            default_colormap=_DISPLAY_COLORMAP[variable_key],
        )

    for variable_key in _FIFTEEN_METRE_VARIABLES:
        if variable_key not in standardised.data_vars or Dimension.DEPTH.key() not in standardised[variable_key].dims:
            continue
        layer_key = f"{variable_key}_15m"
        layers[layer_key] = _fifteen_metre_layer(standardised, variable_key)
        specs[layer_key] = VariableSpec(
            standard_name=variable_key,
            depth=_FIFTEEN_METRE_DEPTH_LABEL,
            units=VARIABLE_METADATA[variable_key][1],
            default_colormap=_DISPLAY_COLORMAP[variable_key],
        )

    return xarray.Dataset(layers), specs
