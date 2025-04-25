# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from functools import partial
from typing import Iterable, List

import numpy
import xarray
import pandas

from oceanbench.core.climate_forecast_standard_names import (
    remane_dataset_with_standard_names,
)
from oceanbench.core.dataset_utils import (
    Variable,
    Dimension,
    DepthLevel,
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


DEPTH_LABELS: dict[DepthLevel, str] = {
    DepthLevel.SURFACE: "surface",
    DepthLevel.MINUS_50_METERS: "50m",
    DepthLevel.MINUS_200_METERS: "200m",
    DepthLevel.MINUS_550_METERS: "550m",
}


LEAD_DAYS_COUNT = 10


def _harmonised_dataset(dataset: xarray.Dataset, variables: list[Variable]) -> xarray.Dataset:
    standard_dataset = remane_dataset_with_standard_names(dataset)
    dataset_with_lead_day_labels = standard_dataset.assign({Dimension.TIME.key(): list(range(LEAD_DAYS_COUNT))})
    dataset_with_depth_selected = dataset_with_lead_day_labels.sel(
        {Dimension.DEPTH.key(): [depth_level.value for depth_level in DepthLevel]}
    )
    dataset_with_depth_labels = dataset_with_depth_selected.assign(
        {Dimension.DEPTH.key(): [DEPTH_LABELS[depth_level] for depth_level in DepthLevel]}
    )
    dataset_with_variable_selected = dataset_with_depth_labels[[variable.key() for variable in variables]]
    return dataset_with_variable_selected


def _rmsd(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
) -> xarray.Dataset:
    return numpy.sqrt(
        ((challenger_dataset - reference_dataset) ** 2).mean(dim=[Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()])
    )


def _mean_of_all_datasets(
    datasets: Iterable[xarray.Dataset],
) -> xarray.Dataset:
    return xarray.concat(datasets, dim="ensemble").mean(dim="ensemble")


def _has_depths(dataset: xarray.Dataset, variable_name: str) -> bool:
    return Dimension.DEPTH.key() in dataset[variable_name].coords


def _variale_depth_label(dataset: xarray.Dataset, variable: str, depth_label: str) -> str:
    return (
        f"{depth_label} {VARIABLE_LABELS[variable]}" if _has_depths(dataset, variable) else VARIABLE_LABELS[variable]
    ).capitalize()


def _to_pretty_dataframe(dataset: xarray.Dataset, variables: list[Variable]) -> pandas.DataFrame:
    indexes_of_variables_sorted: list[tuple[str, str]] = [
        (depth_level, variable.key()) for depth_level in DEPTH_LABELS.values() for variable in variables
    ]
    indexes_of_variables_without_depth: list[tuple[str, str]] = [
        (depth_level, variable.key())
        for depth_level in DEPTH_LABELS.values()
        for variable in variables
        if depth_level != DEPTH_LABELS[DepthLevel.SURFACE] and not _has_depths(dataset, variable.key())
    ]
    dataframe_3d: pandas.DataFrame = (
        dataset.to_dataframe()
        .reset_index()
        .pivot(index=[Dimension.TIME.key()], columns=[Dimension.DEPTH.key()])
        .set_index([lead_day_labels(1, LEAD_DAYS_COUNT)])
        .T.rename_axis(columns=None)
        .swaplevel(0, 1)
        .set_index([indexes_of_variables_sorted])
        .drop(index=indexes_of_variables_without_depth)
    )
    return dataframe_3d.set_index(
        dataframe_3d.index.map(
            lambda depth_variable: _variale_depth_label(dataset, depth_variable[1], depth_variable[0])
        )
    )


def rmsd(
    challenger_datasets: List[xarray.Dataset],
    reference_datasets: List[xarray.Dataset],
    variables: List[Variable],
) -> pandas.DataFrame:
    harmonise = partial(_harmonised_dataset, variables=variables)
    harmonised_challenger_datasets = map(harmonise, challenger_datasets)
    harmonised_reference_datasets = map(harmonise, reference_datasets)
    rmsds = map(_rmsd, harmonised_challenger_datasets, harmonised_reference_datasets)
    return _to_pretty_dataframe(_mean_of_all_datasets(rmsds), variables)
