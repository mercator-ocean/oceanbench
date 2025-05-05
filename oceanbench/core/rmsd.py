# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray
import pandas

from oceanbench.core.dataset_utils import (
    DEPTH_LABELS,
    LEAD_DAYS_COUNT,
    Variable,
    Dimension,
    DepthLevel,
    assign_depth_dimension,
)
from oceanbench.core.lead_day_utils import lead_day_labels

VARIABLE_LABELS: dict[str, str] = {
    Variable.HEIGHT.key(): "surface height",
    Variable.TEMPERATURE.key(): "temperature",
    Variable.SALINITY.key(): "salinity",
    Variable.NORTHWARD_VELOCITY.key(): "northward velocity",
    Variable.EASTWARD_VELOCITY.key(): "eastward velocity",
    Variable.MIXED_LAYER_DEPTH.key(): "mixed layer depth",
    Variable.NORTHWARD_GEOSTROPHIC_VELOCITY.key(): "northward geostrophic velocity",
    Variable.EASTWARD_GEOSTROPHIC_VELOCITY.key(): "eastward geostrophic velocity",
}


def _rmsd(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
) -> xarray.Dataset:
    return numpy.sqrt(
        ((challenger_dataset - reference_dataset) ** 2).mean(dim=[Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()])
    ).mean(dim="start_datetime")


def _has_depths(dataset: xarray.Dataset, variable_name: str) -> bool:
    return Dimension.DEPTH.key() in dataset[variable_name].coords


def _variable_depth_label(dataset: xarray.Dataset, variable: str, depth_label: str) -> str:
    return (
        f"{depth_label} {VARIABLE_LABELS[variable]}" if _has_depths(dataset, variable) else VARIABLE_LABELS[variable]
    ).capitalize()


def _select_dataset_variable_and_depth(dataset: xarray.Dataset, variable_name: str, depth_level: str) -> numpy.ndarray:
    return (
        dataset[variable_name].sel({Dimension.DEPTH.key(): depth_level}).values
        if _has_depths(dataset, variable_name)
        else dataset[variable_name].values
    )


def _to_pretty_dataframe(dataset: xarray.Dataset, variables: list[Variable]) -> pandas.DataFrame:
    dataset_with_depth = assign_depth_dimension(dataset) if dataset.get(Dimension.DEPTH.key()) is None else dataset
    values_2d: dict[str, numpy.ndarray] = {
        _variable_depth_label(dataset_with_depth, variable.key(), depth_level): _select_dataset_variable_and_depth(
            dataset_with_depth, variable.key(), depth_level
        )
        for depth_level in DEPTH_LABELS.values()
        for variable in variables
        if depth_level == DEPTH_LABELS[DepthLevel.SURFACE] or _has_depths(dataset_with_depth, variable.key())
    }
    return pandas.DataFrame(values_2d).set_index([lead_day_labels(1, LEAD_DAYS_COUNT)]).T


def _select_variables(dataset: xarray.Dataset, variables: list[Variable]) -> xarray.Dataset:
    return dataset[[variable.key() for variable in variables]]


def rmsd(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
) -> pandas.DataFrame:
    return _to_pretty_dataframe(
        _rmsd(
            _select_variables(challenger_dataset, variables),
            _select_variables(reference_dataset, variables),
        ),
        variables,
    )
