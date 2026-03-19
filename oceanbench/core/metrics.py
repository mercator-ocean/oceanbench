# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
import xarray

from oceanbench.core.dataset_utils import Variable
from oceanbench.core.derived_quantities import compute_mixed_layer_depth
from oceanbench.core.derived_quantities import compute_geostrophic_currents
from oceanbench.core.instrumentation import instrumented_operation, log_event
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.rmsd import rmsd
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.classIV import rmsd_class4_validation
from oceanbench.core.references.observations import observations
from oceanbench.core.remote_http import with_remote_http_retries

from oceanbench.core.lagrangian_trajectory import deviation_of_lagrangian_trajectories

OBSERVATIONS_UNAVAILABLE_ERROR_PREFIX = "OBSERVATIONS_NOT_AVAILABLE:"
_REFERENCE_DATASET_CACHE: dict[tuple[str, int], xarray.Dataset] = {}


def _run_metric(metric_name: str, callback):
    with instrumented_operation("metric", metric=metric_name):
        return callback()


def _cached_reference_dataset(
    dataset_name: str,
    challenger_dataset: xarray.Dataset,
    load_reference_dataset,
) -> xarray.Dataset:
    cache_key = (dataset_name, id(challenger_dataset))
    cached_dataset = _REFERENCE_DATASET_CACHE.get(cache_key)
    if cached_dataset is not None:
        log_event("reference_dataset_cache_hit", dataset=dataset_name)
        return cached_dataset

    log_event("reference_dataset_cache_miss", dataset=dataset_name)
    reference_dataset = load_reference_dataset(challenger_dataset)
    _REFERENCE_DATASET_CACHE[cache_key] = reference_dataset
    return reference_dataset


def _glorys_reference_dataset(challenger_dataset: xarray.Dataset) -> xarray.Dataset:
    return _cached_reference_dataset("glorys", challenger_dataset, glorys_reanalysis_dataset)


def _glo12_reference_dataset(challenger_dataset: xarray.Dataset) -> xarray.Dataset:
    return _cached_reference_dataset("glo12", challenger_dataset, glo12_analysis_dataset)


def rmsd_of_variables_compared_to_observations(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    def compute():
        try:
            observation_dataset = observations(challenger_dataset)
        except ValueError as error:
            error_message = str(error)
            if error_message.startswith(OBSERVATIONS_UNAVAILABLE_ERROR_PREFIX):
                return pandas.DataFrame(
                    {"Message": [error_message.replace(OBSERVATIONS_UNAVAILABLE_ERROR_PREFIX, "", 1).strip()]}
                )
            raise
        return rmsd_class4_validation(
            challenger_dataset=challenger_dataset,
            reference_dataset=observation_dataset,
            variables=[
                Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
                Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
                Variable.SEA_WATER_SALINITY,
                Variable.NORTHWARD_SEA_WATER_VELOCITY,
                Variable.EASTWARD_SEA_WATER_VELOCITY,
            ],
        )

    return _run_metric(
        "observation-based RMSD",
        lambda: with_remote_http_retries("observation-based RMSD", compute),
    )


def rmsd_of_variables_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return _run_metric(
        "GLORYS variable RMSD",
        lambda: with_remote_http_retries(
            "GLORYS variable RMSD",
            lambda: rmsd(
                challenger_dataset=challenger_dataset,
                reference_dataset=_glorys_reference_dataset(challenger_dataset),
                variables=[
                    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
                    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
                    Variable.SEA_WATER_SALINITY,
                    Variable.NORTHWARD_SEA_WATER_VELOCITY,
                    Variable.EASTWARD_SEA_WATER_VELOCITY,
                ],
            ),
        ),
    )


def rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return _run_metric(
        "GLORYS mixed layer depth RMSD",
        lambda: with_remote_http_retries(
            "GLORYS mixed layer depth RMSD",
            lambda: rmsd(
                challenger_dataset=compute_mixed_layer_depth(challenger_dataset),
                reference_dataset=compute_mixed_layer_depth(_glorys_reference_dataset(challenger_dataset)),
                variables=[
                    Variable.MIXED_LAYER_DEPTH,
                ],
            ),
        ),
    )


def rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return _run_metric(
        "GLORYS geostrophic current RMSD",
        lambda: with_remote_http_retries(
            "GLORYS geostrophic current RMSD",
            lambda: rmsd(
                challenger_dataset=compute_geostrophic_currents(challenger_dataset),
                reference_dataset=compute_geostrophic_currents(_glorys_reference_dataset(challenger_dataset)),
                variables=[
                    Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,
                    Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,
                ],
            ),
        ),
    )


def deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return _run_metric(
        "GLORYS lagrangian trajectories",
        lambda: with_remote_http_retries(
            "GLORYS lagrangian trajectories",
            lambda: deviation_of_lagrangian_trajectories(
                challenger_dataset=challenger_dataset,
                reference_dataset=_glorys_reference_dataset(challenger_dataset),
            ),
        ),
    )


def rmsd_of_variables_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return _run_metric(
        "GLO12 variable RMSD",
        lambda: with_remote_http_retries(
            "GLO12 variable RMSD",
            lambda: rmsd(
                challenger_dataset=challenger_dataset,
                reference_dataset=_glo12_reference_dataset(challenger_dataset),
                variables=[
                    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
                    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
                    Variable.SEA_WATER_SALINITY,
                    Variable.NORTHWARD_SEA_WATER_VELOCITY,
                    Variable.EASTWARD_SEA_WATER_VELOCITY,
                ],
            ),
        ),
    )


def rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return _run_metric(
        "GLO12 mixed layer depth RMSD",
        lambda: with_remote_http_retries(
            "GLO12 mixed layer depth RMSD",
            lambda: rmsd(
                challenger_dataset=compute_mixed_layer_depth(challenger_dataset),
                reference_dataset=compute_mixed_layer_depth(_glo12_reference_dataset(challenger_dataset)),
                variables=[
                    Variable.MIXED_LAYER_DEPTH,
                ],
            ),
        ),
    )


def rmsd_of_geostrophic_currents_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return _run_metric(
        "GLO12 geostrophic current RMSD",
        lambda: with_remote_http_retries(
            "GLO12 geostrophic current RMSD",
            lambda: rmsd(
                challenger_dataset=compute_geostrophic_currents(challenger_dataset),
                reference_dataset=compute_geostrophic_currents(_glo12_reference_dataset(challenger_dataset)),
                variables=[
                    Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,
                    Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,
                ],
            ),
        ),
    )


def deviation_of_lagrangian_trajectories_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return _run_metric(
        "GLO12 lagrangian trajectories",
        lambda: with_remote_http_retries(
            "GLO12 lagrangian trajectories",
            lambda: deviation_of_lagrangian_trajectories(
                challenger_dataset=challenger_dataset,
                reference_dataset=_glo12_reference_dataset(challenger_dataset),
            ),
        ),
    )
