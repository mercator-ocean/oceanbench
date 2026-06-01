# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass
from functools import cached_property
import warnings

import pandas
import xarray
from dask.array.core import PerformanceWarning
from IPython.display import HTML

from oceanbench.core import visualization
from oceanbench.core.classIV import class4_validation_dataframe, rmsd_class4_validation_dataframe
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.derived_quantities import compute_geostrophic_currents, compute_mixed_layer_depth
from oceanbench.core.lagrangian_trajectory import (
    deviation_of_lagrangian_trajectories,
    lagrangian_particle_count_for_region,
)
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.references.observations import ObservationDataUnavailableError, observations
from oceanbench.core.regions import GLOBAL_REGION_NAME, RegionLike, subset_dataset_to_region
from oceanbench.core.rmsd import rmsd

GLORYS_REFERENCE_NAME = "GLORYS reanalysis"
GLO12_REFERENCE_NAME = "GLO12 analysis"
GLOBAL_LAGRANGIAN_PARTICLE_COUNT = 10000
MINIMUM_LAGRANGIAN_PARTICLE_COUNT = 2000
DISPLAY_LAGRANGIAN_PARTICLE_COUNT = 1000

FORECAST_COMPARISON_VARIABLES = [
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
    Variable.SEA_WATER_SALINITY,
    Variable.EASTWARD_SEA_WATER_VELOCITY,
    Variable.NORTHWARD_SEA_WATER_VELOCITY,
]
DYNAMIC_DIAGNOSTIC_VARIABLES = [
    Variable.MIXED_LAYER_DEPTH,
    Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,
    Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,
]
GEOSTROPHIC_CURRENT_VARIABLES = [
    Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,
    Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,
]
ZONAL_PSD_VARIABLES = [
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
]


@dataclass
class Class4ObservationReport:
    dataset: xarray.Dataset | None
    comparison_dataframe: pandas.DataFrame | None
    rmsd: pandas.DataFrame

    @property
    def has_comparison_dataframe(self) -> bool:
        return self.comparison_dataframe is not None and not self.comparison_dataframe.empty


@dataclass
class EvaluationReportContext:
    challenger_dataset: xarray.Dataset
    region: RegionLike = GLOBAL_REGION_NAME

    @cached_property
    def regional_challenger_dataset(self) -> xarray.Dataset:
        return subset_dataset_to_region(self.challenger_dataset, self.region)

    @cached_property
    def glorys_dataset(self) -> xarray.Dataset:
        return subset_dataset_to_region(glorys_reanalysis_dataset(self.regional_challenger_dataset), self.region)

    @cached_property
    def glo12_dataset(self) -> xarray.Dataset:
        return subset_dataset_to_region(glo12_analysis_dataset(self.regional_challenger_dataset), self.region)

    @cached_property
    def reference_datasets(self) -> dict[str, xarray.Dataset]:
        return {
            GLORYS_REFERENCE_NAME: self.glorys_dataset,
            GLO12_REFERENCE_NAME: self.glo12_dataset,
        }

    @cached_property
    def challenger_mixed_layer_depth_dataset(self) -> xarray.Dataset:
        return compute_mixed_layer_depth(self.regional_challenger_dataset)

    @cached_property
    def glorys_mixed_layer_depth_dataset(self) -> xarray.Dataset:
        return compute_mixed_layer_depth(self.glorys_dataset)

    @cached_property
    def glo12_mixed_layer_depth_dataset(self) -> xarray.Dataset:
        return compute_mixed_layer_depth(self.glo12_dataset)

    @cached_property
    def challenger_geostrophic_current_dataset(self) -> xarray.Dataset:
        return _compute_geostrophic_currents(self.regional_challenger_dataset)

    @cached_property
    def glorys_geostrophic_current_dataset(self) -> xarray.Dataset:
        return _compute_geostrophic_currents(self.glorys_dataset)

    @cached_property
    def glo12_geostrophic_current_dataset(self) -> xarray.Dataset:
        return _compute_geostrophic_currents(self.glo12_dataset)

    @cached_property
    def challenger_dynamic_dataset(self) -> xarray.Dataset:
        return xarray.merge([self.challenger_mixed_layer_depth_dataset, self.challenger_geostrophic_current_dataset])

    @cached_property
    def glorys_dynamic_dataset(self) -> xarray.Dataset:
        return xarray.merge([self.glorys_mixed_layer_depth_dataset, self.glorys_geostrophic_current_dataset])

    @cached_property
    def glo12_dynamic_dataset(self) -> xarray.Dataset:
        return xarray.merge([self.glo12_mixed_layer_depth_dataset, self.glo12_geostrophic_current_dataset])

    @cached_property
    def dynamic_reference_datasets(self) -> dict[str, xarray.Dataset]:
        return {
            GLORYS_REFERENCE_NAME: self.glorys_dynamic_dataset,
            GLO12_REFERENCE_NAME: self.glo12_dynamic_dataset,
        }

    @cached_property
    def glorys_variable_rmsd(self) -> pandas.DataFrame:
        return _rmsd(self.regional_challenger_dataset, self.glorys_dataset, FORECAST_COMPARISON_VARIABLES)

    @cached_property
    def glorys_mixed_layer_depth_rmsd(self) -> pandas.DataFrame:
        return _rmsd(
            self.challenger_mixed_layer_depth_dataset,
            self.glorys_mixed_layer_depth_dataset,
            [Variable.MIXED_LAYER_DEPTH],
        )

    @cached_property
    def glorys_geostrophic_current_rmsd(self) -> pandas.DataFrame:
        return _rmsd(
            self.challenger_geostrophic_current_dataset,
            self.glorys_geostrophic_current_dataset,
            GEOSTROPHIC_CURRENT_VARIABLES,
        )

    @cached_property
    def glorys_lagrangian_trajectory_deviation(self) -> pandas.DataFrame:
        return deviation_of_lagrangian_trajectories(
            challenger_dataset=self.regional_challenger_dataset,
            reference_dataset=self.glorys_dataset,
            particle_count=self.lagrangian_particle_count,
        )

    @cached_property
    def glo12_variable_rmsd(self) -> pandas.DataFrame:
        return _rmsd(self.regional_challenger_dataset, self.glo12_dataset, FORECAST_COMPARISON_VARIABLES)

    @cached_property
    def glo12_mixed_layer_depth_rmsd(self) -> pandas.DataFrame:
        return _rmsd(
            self.challenger_mixed_layer_depth_dataset,
            self.glo12_mixed_layer_depth_dataset,
            [Variable.MIXED_LAYER_DEPTH],
        )

    @cached_property
    def glo12_geostrophic_current_rmsd(self) -> pandas.DataFrame:
        return _rmsd(
            self.challenger_geostrophic_current_dataset,
            self.glo12_geostrophic_current_dataset,
            GEOSTROPHIC_CURRENT_VARIABLES,
        )

    @cached_property
    def glo12_lagrangian_trajectory_deviation(self) -> pandas.DataFrame:
        return deviation_of_lagrangian_trajectories(
            challenger_dataset=self.regional_challenger_dataset,
            reference_dataset=self.glo12_dataset,
            particle_count=self.lagrangian_particle_count,
        )

    @cached_property
    def lagrangian_particle_count(self) -> int:
        return lagrangian_particle_count_for_region(
            self.challenger_dataset,
            self.regional_challenger_dataset,
            global_particle_count=GLOBAL_LAGRANGIAN_PARTICLE_COUNT,
            minimum_particle_count=MINIMUM_LAGRANGIAN_PARTICLE_COUNT,
        )

    @cached_property
    def class4_observation(self) -> Class4ObservationReport:
        try:
            observation_dataset = subset_dataset_to_region(observations(self.regional_challenger_dataset), self.region)
        except ObservationDataUnavailableError as error:
            return Class4ObservationReport(
                dataset=None,
                comparison_dataframe=None,
                rmsd=pandas.DataFrame({"Message": [str(error)]}),
            )

        comparison_dataframe = class4_validation_dataframe(
            challenger_dataset=self.regional_challenger_dataset,
            reference_dataset=observation_dataset,
            variables=FORECAST_COMPARISON_VARIABLES,
        )
        return Class4ObservationReport(
            dataset=observation_dataset,
            comparison_dataframe=comparison_dataframe,
            rmsd=rmsd_class4_validation_dataframe(
                comparison_dataframe,
                lead_days_count=self.regional_challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()],
            ),
        )

    @cached_property
    def class4_observation_error_explorer(self) -> HTML | None:
        if not self.class4_observation.has_comparison_dataframe:
            return None
        return visualization.plot_class4_observation_error_explorer(
            challenger_dataset=self.regional_challenger_dataset,
            observation_dataset=self.class4_observation.dataset,
            variables=FORECAST_COMPARISON_VARIABLES,
            title="Class IV observation error maps",
            comparison_dataframe=self.class4_observation.comparison_dataframe,
        )

    @cached_property
    def lagrangian_trajectory_explorer(self) -> HTML:
        return visualization.plot_multi_reference_lagrangian_trajectory_explorer(
            self.regional_challenger_dataset,
            self.reference_datasets,
            particle_count=DISPLAY_LAGRANGIAN_PARTICLE_COUNT,
        )

    @cached_property
    def eddy_matching_explorer(self) -> HTML:
        return visualization.plot_multi_reference_eddy_matching_explorer(
            self.regional_challenger_dataset,
            self.reference_datasets,
        )

    @cached_property
    def forecast_comparison_explorer(self) -> HTML:
        return visualization.plot_multi_reference_surface_comparison_explorer(
            self.regional_challenger_dataset,
            self.reference_datasets,
            variables=FORECAST_COMPARISON_VARIABLES,
            title="Forecast comparison maps",
        )

    @cached_property
    def dynamic_diagnostic_explorer(self) -> HTML:
        return visualization.plot_multi_reference_surface_comparison_explorer(
            self.challenger_dynamic_dataset,
            self.dynamic_reference_datasets,
            variables=DYNAMIC_DIAGNOSTIC_VARIABLES,
            title="Dynamic diagnostic maps",
        )

    @cached_property
    def zonal_psd_explorer(self) -> HTML:
        return visualization.plot_multi_reference_zonal_psd_comparison_explorer(
            self.regional_challenger_dataset,
            self.reference_datasets,
            variables=ZONAL_PSD_VARIABLES,
        )


def _compute_geostrophic_currents(dataset: xarray.Dataset) -> xarray.Dataset:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerformanceWarning, message="Increasing number of chunks.*")
        return compute_geostrophic_currents(dataset)


def _rmsd(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=challenger_dataset,
        reference_dataset=reference_dataset,
        variables=variables,
    )


def prepare_evaluation_report(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
) -> EvaluationReportContext:
    return EvaluationReportContext(
        challenger_dataset=challenger_dataset,
        region=region,
    )


@dataclass
class LiveEvaluationReportContext:
    challenger_dataset: xarray.Dataset
    region: RegionLike = GLOBAL_REGION_NAME
    observation_zarr_template: str | None = None
    observation_last_available_day: str | None = None

    @cached_property
    def regional_challenger_dataset(self) -> xarray.Dataset:
        return subset_dataset_to_region(self.challenger_dataset, self.region)

    @cached_property
    def class4_observation(self) -> Class4ObservationReport:
        try:
            observation_dataset = subset_dataset_to_region(
                observations(
                    self.regional_challenger_dataset,
                    zarr_template=self.observation_zarr_template,
                    last_available_day=self.observation_last_available_day,
                ),
                self.region,
            )
        except ObservationDataUnavailableError as error:
            return Class4ObservationReport(
                dataset=None,
                comparison_dataframe=None,
                rmsd=pandas.DataFrame({"Message": [str(error)]}),
            )

        comparison_dataframe = class4_validation_dataframe(
            challenger_dataset=self.regional_challenger_dataset,
            reference_dataset=observation_dataset,
            variables=FORECAST_COMPARISON_VARIABLES,
        )
        return Class4ObservationReport(
            dataset=observation_dataset,
            comparison_dataframe=comparison_dataframe,
            rmsd=rmsd_class4_validation_dataframe(
                comparison_dataframe,
                lead_days_count=self.regional_challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()],
            ),
        )

    @cached_property
    def class4_observation_error_explorer(self) -> HTML | None:
        if not self.class4_observation.has_comparison_dataframe:
            return None
        return visualization.plot_class4_observation_error_explorer(
            challenger_dataset=self.regional_challenger_dataset,
            observation_dataset=self.class4_observation.dataset,
            variables=FORECAST_COMPARISON_VARIABLES,
            title="Live Class IV observation error maps",
            comparison_dataframe=self.class4_observation.comparison_dataframe,
        )


def prepare_live_evaluation_report(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
    observation_zarr_template: str | None = None,
    observation_last_available_day: str | None = None,
) -> LiveEvaluationReportContext:
    return LiveEvaluationReportContext(
        challenger_dataset=challenger_dataset,
        region=region,
        observation_zarr_template=observation_zarr_template,
        observation_last_available_day=observation_last_available_day,
    )
