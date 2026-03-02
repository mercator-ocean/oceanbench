# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
import xarray

from oceanbench.core.dataset_utils import Variable
from oceanbench.core.derived_quantities import compute_mixed_layer_depth
from oceanbench.core.derived_quantities import compute_geostrophic_currents
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.rmsd import rmsd
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.classIV import rmsd_class4_validation
from oceanbench.core.references.observations import observations

from oceanbench.core.lagrangian_trajectory import (
    deviation_of_lagrangian_trajectories,
)
from oceanbench.core.memory_diagnostics import default_memory_tracker, describe_dataset

OBSERVATIONS_UNAVAILABLE_ERROR_PREFIX = "OBSERVATIONS_NOT_AVAILABLE:"
_memory_tracker = default_memory_tracker("metrics")


def rmsd_of_variables_compared_to_observations(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    with _memory_tracker.step("rmsd_observations"):
        describe_dataset(challenger_dataset, "challenger_dataset", _memory_tracker)
        try:
            with _memory_tracker.step("load_observations"):
                observation_dataset = observations(challenger_dataset)
            describe_dataset(observation_dataset, "observation_dataset", _memory_tracker)
        except ValueError as error:
            error_message = str(error)
            if error_message.startswith(OBSERVATIONS_UNAVAILABLE_ERROR_PREFIX):
                return pandas.DataFrame(
                    {"Message": [error_message.replace(OBSERVATIONS_UNAVAILABLE_ERROR_PREFIX, "", 1).strip()]}
                )
            raise
        with _memory_tracker.step("compute_rmsd_class4_validation"):
            result = rmsd_class4_validation(
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
        return result


def rmsd_of_variables_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    with _memory_tracker.step("rmsd_glorys_variables"):
        describe_dataset(challenger_dataset, "challenger_dataset", _memory_tracker)
        with _memory_tracker.step("load_glorys_reanalysis_dataset"):
            reference_dataset = glorys_reanalysis_dataset(challenger_dataset)
        describe_dataset(reference_dataset, "glorys_reference_dataset", _memory_tracker)
        with _memory_tracker.step("compute_rmsd_glorys_variables"):
            return rmsd(
                challenger_dataset=challenger_dataset,
                reference_dataset=reference_dataset,
                variables=[
                    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
                    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
                    Variable.SEA_WATER_SALINITY,
                    Variable.NORTHWARD_SEA_WATER_VELOCITY,
                    Variable.EASTWARD_SEA_WATER_VELOCITY,
                ],
            )


def rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    with _memory_tracker.step("rmsd_glorys_mixed_layer_depth"):
        with _memory_tracker.step("compute_challenger_mixed_layer_depth"):
            challenger_with_mld = compute_mixed_layer_depth(challenger_dataset)
        describe_dataset(challenger_with_mld, "challenger_with_mld", _memory_tracker)
        with _memory_tracker.step("load_glorys_for_mixed_layer_depth"):
            glorys_dataset = glorys_reanalysis_dataset(challenger_dataset)
        with _memory_tracker.step("compute_glorys_mixed_layer_depth"):
            glorys_with_mld = compute_mixed_layer_depth(glorys_dataset)
        describe_dataset(glorys_with_mld, "glorys_with_mld", _memory_tracker)
        with _memory_tracker.step("compute_rmsd_glorys_mixed_layer_depth"):
            return rmsd(
                challenger_dataset=challenger_with_mld,
                reference_dataset=glorys_with_mld,
                variables=[
                    Variable.MIXED_LAYER_DEPTH,
                ],
            )


def rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    with _memory_tracker.step("rmsd_glorys_geostrophic_currents"):
        with _memory_tracker.step("compute_challenger_geostrophic_currents"):
            challenger_geostrophic = compute_geostrophic_currents(challenger_dataset)
        describe_dataset(challenger_geostrophic, "challenger_geostrophic", _memory_tracker)
        with _memory_tracker.step("load_glorys_for_geostrophic_currents"):
            glorys_dataset = glorys_reanalysis_dataset(challenger_dataset)
        with _memory_tracker.step("compute_glorys_geostrophic_currents"):
            glorys_geostrophic = compute_geostrophic_currents(glorys_dataset)
        describe_dataset(glorys_geostrophic, "glorys_geostrophic", _memory_tracker)
        with _memory_tracker.step("compute_rmsd_glorys_geostrophic_currents"):
            return rmsd(
                challenger_dataset=challenger_geostrophic,
                reference_dataset=glorys_geostrophic,
                variables=[
                    Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,
                    Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,
                ],
            )


def deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    with _memory_tracker.step("lagrangian_deviation_glorys"):
        with _memory_tracker.step("load_glorys_for_lagrangian"):
            glorys_dataset = glorys_reanalysis_dataset(challenger_dataset)
        describe_dataset(glorys_dataset, "glorys_reference_dataset", _memory_tracker)
        with _memory_tracker.step("compute_lagrangian_deviation_glorys"):
            return deviation_of_lagrangian_trajectories(
                challenger_dataset=challenger_dataset,
                reference_dataset=glorys_dataset,
            )


def rmsd_of_variables_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    with _memory_tracker.step("rmsd_glo12_variables"):
        describe_dataset(challenger_dataset, "challenger_dataset", _memory_tracker)
        with _memory_tracker.step("load_glo12_analysis_dataset"):
            reference_dataset = glo12_analysis_dataset(challenger_dataset)
        describe_dataset(reference_dataset, "glo12_reference_dataset", _memory_tracker)
        with _memory_tracker.step("compute_rmsd_glo12_variables"):
            return rmsd(
                challenger_dataset=challenger_dataset,
                reference_dataset=reference_dataset,
                variables=[
                    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
                    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
                    Variable.SEA_WATER_SALINITY,
                    Variable.NORTHWARD_SEA_WATER_VELOCITY,
                    Variable.EASTWARD_SEA_WATER_VELOCITY,
                ],
            )


def rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    with _memory_tracker.step("rmsd_glo12_mixed_layer_depth"):
        with _memory_tracker.step("compute_challenger_mld_for_glo12"):
            challenger_with_mld = compute_mixed_layer_depth(challenger_dataset)
        with _memory_tracker.step("load_glo12_for_mixed_layer_depth"):
            glo12_dataset = glo12_analysis_dataset(challenger_dataset)
        with _memory_tracker.step("compute_glo12_mixed_layer_depth"):
            glo12_with_mld = compute_mixed_layer_depth(glo12_dataset)
        describe_dataset(challenger_with_mld, "challenger_with_mld", _memory_tracker)
        describe_dataset(glo12_with_mld, "glo12_with_mld", _memory_tracker)
        with _memory_tracker.step("compute_rmsd_glo12_mixed_layer_depth"):
            return rmsd(
                challenger_dataset=challenger_with_mld,
                reference_dataset=glo12_with_mld,
                variables=[
                    Variable.MIXED_LAYER_DEPTH,
                ],
            )


def rmsd_of_geostrophic_currents_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    with _memory_tracker.step("rmsd_glo12_geostrophic_currents"):
        with _memory_tracker.step("compute_challenger_geostrophic_for_glo12"):
            challenger_geostrophic = compute_geostrophic_currents(challenger_dataset)
        with _memory_tracker.step("load_glo12_for_geostrophic_currents"):
            glo12_dataset = glo12_analysis_dataset(challenger_dataset)
        with _memory_tracker.step("compute_glo12_geostrophic_currents"):
            glo12_geostrophic = compute_geostrophic_currents(glo12_dataset)
        describe_dataset(challenger_geostrophic, "challenger_geostrophic", _memory_tracker)
        describe_dataset(glo12_geostrophic, "glo12_geostrophic", _memory_tracker)
        with _memory_tracker.step("compute_rmsd_glo12_geostrophic_currents"):
            return rmsd(
                challenger_dataset=challenger_geostrophic,
                reference_dataset=glo12_geostrophic,
                variables=[
                    Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,
                    Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,
                ],
            )


def deviation_of_lagrangian_trajectories_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    with _memory_tracker.step("lagrangian_deviation_glo12"):
        with _memory_tracker.step("load_glo12_for_lagrangian"):
            glo12_dataset = glo12_analysis_dataset(challenger_dataset)
        describe_dataset(glo12_dataset, "glo12_reference_dataset", _memory_tracker)
        with _memory_tracker.step("compute_lagrangian_deviation_glo12"):
            return deviation_of_lagrangian_trajectories(
                challenger_dataset=challenger_dataset,
                reference_dataset=glo12_dataset,
            )
