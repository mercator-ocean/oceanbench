# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from functools import partial
import multiprocessing
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
from multiprocessing import Pool

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
    dataset_with_depth_labels = _assign_depth_dimension(dataset_with_depth_selected)
    dataset_with_variable_selected = dataset_with_depth_labels[[variable.key() for variable in variables]]
    return dataset_with_variable_selected


def _rmsd(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
) -> xarray.Dataset:
    print(f"Computing RMSD on {multiprocessing.current_process()}...")
    toto = numpy.sqrt(
        ((challenger_dataset - reference_dataset) ** 2).mean(dim=[Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()])
    )
    print(f"RMSD on {multiprocessing.current_process()} computed")
    return toto


def _mean_of_all_datasets(
    datasets: Iterable[xarray.Dataset],
) -> xarray.Dataset:
    return xarray.concat(datasets, dim="ensemble").mean(dim="ensemble")


def _has_depths(dataset: xarray.Dataset, variable_name: str) -> bool:
    return Dimension.DEPTH.key() in dataset[variable_name].coords


def _variable_depth_label(dataset: xarray.Dataset, variable: str, depth_label: str) -> str:
    return (
        f"{depth_label} {VARIABLE_LABELS[variable]}" if _has_depths(dataset, variable) else VARIABLE_LABELS[variable]
    ).capitalize()


def _assign_depth_dimension(dataset: xarray.Dataset) -> xarray.Dataset:
    return dataset.assign({Dimension.DEPTH.key(): [DEPTH_LABELS[depth_level] for depth_level in DepthLevel]})


def _select_dataset_variable_and_depth(dataset: xarray.Dataset, variable_name: str, depth_level: str) -> numpy.ndarray:
    return (
        dataset[variable_name].sel({Dimension.DEPTH.key(): depth_level}).values
        if _has_depths(dataset, variable_name)
        else dataset[variable_name].values
    )


def _to_pretty_dataframe(dataset: xarray.Dataset, variables: list[Variable]) -> pandas.DataFrame:
    dataset_with_depth = _assign_depth_dimension(dataset) if dataset.get(Dimension.DEPTH.key()) is None else dataset
    values_2d: dict[str, numpy.ndarray] = {
        _variable_depth_label(dataset_with_depth, variable.key(), depth_level): _select_dataset_variable_and_depth(
            dataset_with_depth, variable.key(), depth_level
        )
        for depth_level in DEPTH_LABELS.values()
        for variable in variables
        if depth_level == DEPTH_LABELS[DepthLevel.SURFACE] or _has_depths(dataset_with_depth, variable.key())
    }
    return pandas.DataFrame(values_2d).set_index([lead_day_labels(1, LEAD_DAYS_COUNT)]).T


def rmsd(
    challenger_datasets: List[xarray.Dataset],
    reference_datasets: List[xarray.Dataset],
    variables: List[Variable],
) -> pandas.DataFrame:
    harmonise = partial(_harmonised_dataset, variables=variables)
    print("Harmonizing...")
    harmonised_challenger_datasets = list(map(harmonise, challenger_datasets))
    harmonised_reference_datasets = list(map(harmonise, reference_datasets))
    print("Harmonized")
    with Pool(processes=4) as pool:
        rmsds = pool.starmap(
            _rmsd,
            zip(harmonised_challenger_datasets, harmonised_reference_datasets),
        )
        return _to_pretty_dataframe(_mean_of_all_datasets(rmsds), variables)
