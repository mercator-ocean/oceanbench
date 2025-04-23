# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from functools import partial
import multiprocessing
from typing import List

import numpy
import xarray
import pandas

from itertools import product

from oceanbench.core.dataset_utils import (
    Variable,
    Dimension,
    DepthLevel,
    get_variable,
    select_variable_day_and_depth,
)
from oceanbench.core.lead_day_utils import lead_day_labels


VARIABLE_LABELS = {
    Variable.HEIGHT: "surface height",
    Variable.TEMPERATURE: "temperature",
    Variable.SALINITY: "salinity",
    Variable.NORTHWARD_VELOCITY: "northward velocity",
    Variable.EASTWARD_VELOCITY: "eastward velocity",
    Variable.MIXED_LAYER_DEPTH: "mixed layer depth",
    Variable.NORTHWARD_GEOSTROPHIC_VELOCITY: "northward geostrophic velocity",
    Variable.EASTWARD_GEOSTROPHIC_VELOCITY: "eastward geostrophic velocity",
}


DEPTH_LABELS: dict[DepthLevel, str] = {
    DepthLevel.SURFACE: "surface",
    DepthLevel.MINUS_50_METERS: "50m",
    DepthLevel.MINUS_200_METERS: "200m",
    DepthLevel.MINUS_550_METERS: "550m",
}


def _rmsd(data, reference_data):
    mask = ~numpy.isnan(data) & ~numpy.isnan(reference_data)
    rmsd = numpy.sqrt(numpy.mean((data[mask] - reference_data[mask]) ** 2))
    return rmsd


def _get_rmsd(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variable: Variable,
    depth_level: DepthLevel,
    lead_day: int,
) -> float:
    cpu_count = multiprocessing.cpu_count()
    with multiprocessing.Pool(cpu_count) as _:
        challenger_dataarray = select_variable_day_and_depth(challenger_dataset, variable, depth_level, lead_day)
        reference_dataarray = select_variable_day_and_depth(reference_dataset, variable, depth_level, lead_day)
        return _rmsd(challenger_dataarray.data, reference_dataarray.data)


LEAD_DAYS_COUNT = 10


def _get_rmsd_for_all_lead_days(
    dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variable: Variable,
    depth_level: DepthLevel,
) -> list[float]:
    return list(
        map(
            partial(
                _get_rmsd,
                dataset,
                reference_dataset,
                variable,
                depth_level,
            ),
            range(LEAD_DAYS_COUNT),
        )
    )


def _compute_rmsd(
    datasets: List[xarray.Dataset],
    reference_datasets: List[xarray.Dataset],
    variable: Variable,
    depth_level: DepthLevel,
) -> numpy.ndarray:

    all_rmsd = numpy.array(
        list(
            map(
                partial(
                    _get_rmsd_for_all_lead_days,
                    variable=variable,
                    depth_level=depth_level,
                ),
                datasets,
                reference_datasets,
            )
        )
    )
    return all_rmsd.mean(axis=0)


def _variale_depth_label(dataset: xarray.Dataset, variable: Variable, depth_level: DepthLevel) -> str:
    return (
        f"{DEPTH_LABELS[depth_level]} {VARIABLE_LABELS[variable]}"
        if _has_depths(dataset, variable)
        else VARIABLE_LABELS[variable]
    ).capitalize()


def _has_depths(dataset: xarray.Dataset, variable: Variable) -> bool:
    return Dimension.DEPTH.dimension_name_from_dataset(dataset) in get_variable(dataset, variable).coords


def _is_surface(depth_level: DepthLevel) -> bool:
    return depth_level == DepthLevel.SURFACE


def _variable_and_depth_combinations(
    dataset: xarray.Dataset, variables: list[Variable]
) -> list[tuple[Variable, DepthLevel]]:
    return list(
        (variable, depth_level)
        for (depth_level, variable) in product(list(DepthLevel), variables)
        if (_has_depths(dataset, variable) or _is_surface(depth_level))
    )


def rmsd(
    challenger_datasets: List[xarray.Dataset],
    reference_datasets: List[xarray.Dataset],
    variables: List[Variable],
) -> pandas.DataFrame:
    all_combinations = _variable_and_depth_combinations(challenger_datasets[0], variables)
    scores = {
        _variale_depth_label(challenger_datasets[0], variable, depth_level): list(
            _compute_rmsd(
                challenger_datasets,
                reference_datasets,
                variable,
                depth_level,
            )
        )
        for (variable, depth_level) in all_combinations
    }
    score_dataframe = pandas.DataFrame(scores)
    score_dataframe.index = lead_day_labels(1, LEAD_DAYS_COUNT)
    return score_dataframe.T
