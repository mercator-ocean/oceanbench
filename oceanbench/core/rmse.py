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


VARIABLE_LABELS = {
    Variable.HEIGHT: "surface height",
    Variable.TEMPERATURE: "temperature",
    Variable.SALINITY: "salinity",
    Variable.NORTHWARD_VELOCITY: "northward velocity",
    Variable.EASTWARD_VELOCITY: "eastward velocity",
    Variable.MIXED_LAYER_DEPTH: "mixed layer depth",
}


DEPTH_LABELS: dict[DepthLevel, str] = {
    DepthLevel.SURFACE: "surface",
    DepthLevel.MINUS_50_METERS: "50m",
    DepthLevel.MINUS_200_METERS: "200m",
    DepthLevel.MINUS_550_METERS: "550m",
}


def _rmse(data, reference_data):
    mask = ~numpy.isnan(data) & ~numpy.isnan(reference_data)
    rmse = numpy.sqrt(numpy.mean((data[mask] - reference_data[mask]) ** 2))
    return rmse


def _get_rmse(
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
        return _rmse(challenger_dataarray.data, reference_dataarray.data)


LEAD_DAYS_COUNT = 10


def _get_rmse_for_all_lead_days(
    dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variable: Variable,
    depth_level: DepthLevel,
) -> list[float]:
    return list(
        map(
            partial(
                _get_rmse,
                dataset,
                reference_dataset,
                variable,
                depth_level,
            ),
            range(LEAD_DAYS_COUNT),
        )
    )


def _compute_rmse(
    datasets: List[xarray.Dataset],
    reference_datasets: List[xarray.Dataset],
    variable: Variable,
    depth_level: DepthLevel,
) -> numpy.ndarray:

    all_rmse = numpy.array(
        [
            _get_rmse_for_all_lead_days(dataset, reference_dataset, variable, depth_level)
            for dataset, reference_dataset in zip(datasets, reference_datasets)
        ]
    )
    return all_rmse.mean(axis=0)


def _lead_day_labels(day_count) -> list[str]:
    return list(
        map(
            lambda day_index: f"Lead day {day_index}",
            range(1, day_count + 1),
        )
    )


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


def rmse(
    challenger_datasets: List[xarray.Dataset],
    reference_datasets: List[xarray.Dataset],
    variables: List[Variable],
) -> pandas.DataFrame:
    all_combinations = _variable_and_depth_combinations(challenger_datasets[0], variables)
    scores = {
        _variale_depth_label(challenger_datasets[0], variable, depth_level): list(
            _compute_rmse(
                challenger_datasets,
                reference_datasets,
                variable,
                depth_level,
            )
        )
        for (variable, depth_level) in all_combinations
    }
    score_dataframe = pandas.DataFrame(scores)
    score_dataframe.index = _lead_day_labels(len(score_dataframe))
    return score_dataframe.T
