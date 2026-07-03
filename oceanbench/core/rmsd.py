# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray
import pandas

from oceanbench.core.climate_forecast_standard_names import (
    rename_dataset_with_standard_names,
)
from oceanbench.core.dataset_utils import (
    Variable,
    Dimension,
    DepthLevel,
    VARIABLE_METADATA,
)
from oceanbench.core.lead_day_utils import lead_day_labels

DEPTH_LABELS: dict[DepthLevel, str] = {
    DepthLevel.SURFACE: "surface",
    DepthLevel.MINUS_50_METERS: "50m",
    DepthLevel.MINUS_100_METERS: "100m",
    DepthLevel.MINUS_200_METERS: "200m",
    DepthLevel.MINUS_300_METERS: "300m",
    DepthLevel.MINUS_500_METERS: "500m",
}


def _assign_depth_dimension(dataset: xarray.Dataset) -> xarray.Dataset:
    return dataset.assign({Dimension.DEPTH.key(): [DEPTH_LABELS[depth_level] for depth_level in DepthLevel]})


def _spatial_area_weights(dataset: xarray.Dataset) -> xarray.DataArray:
    return numpy.cos(numpy.deg2rad(dataset[Dimension.LATITUDE.key()]))


def _root_mean_squared_error_per_start(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    area_weighted: bool = True,
) -> xarray.Dataset:
    squared_error = (challenger_dataset - reference_dataset) ** 2
    spatial_dimensions = [Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()]
    mean_squared_error = (
        squared_error.weighted(_spatial_area_weights(squared_error)).mean(dim=spatial_dimensions)
        if area_weighted
        else squared_error.mean(dim=spatial_dimensions)
    )
    return numpy.sqrt(mean_squared_error)


def _rmsd(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    area_weighted: bool = True,
) -> xarray.Dataset:
    return _root_mean_squared_error_per_start(challenger_dataset, reference_dataset, area_weighted).mean(
        dim=Dimension.FIRST_DAY_DATETIME.key()
    )


def _has_depths(dataset: xarray.Dataset, variable_name: str) -> bool:
    return Dimension.DEPTH.key() in dataset[variable_name].coords


def _variable_depth_label(dataset: xarray.Dataset, variable: str, depth_label: str) -> str:
    display_name, unit = VARIABLE_METADATA[variable]
    return f"{display_name.capitalize()} ({unit}) [{variable}]{{{depth_label}}}"


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
    lead_days_count = dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    return pandas.DataFrame(values_2d).set_index([lead_day_labels(1, lead_days_count)]).T


def _harmonise_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    standard_dataset = rename_dataset_with_standard_names(dataset)
    lead_days_count = standard_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    dataset_with_lead_day_labels = standard_dataset.assign(
        {Dimension.LEAD_DAY_INDEX.key(): list(range(lead_days_count))}
    )

    dataset_with_depth_selected = dataset_with_lead_day_labels.sel(
        {Dimension.DEPTH.key(): [depth_level.value for depth_level in DepthLevel]}, method="nearest"
    )
    dataset_with_depth_labels = _assign_depth_dimension(dataset_with_depth_selected)
    return dataset_with_depth_labels


def _select_variables(dataset: xarray.Dataset, variables: list[Variable]) -> xarray.Dataset:
    return dataset[[variable.key() for variable in variables]]


def rmsd(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
    area_weighted: bool = True,
) -> pandas.DataFrame:
    prepared_challenger_dataset = _select_variables(_harmonise_dataset(challenger_dataset), variables)
    prepared_reference_dataset = _select_variables(_harmonise_dataset(reference_dataset), variables)
    rmsd_dataset = _rmsd(prepared_challenger_dataset, prepared_reference_dataset, area_weighted)
    computed_rmsd_dataset = rmsd_dataset.compute()
    return _to_pretty_dataframe(computed_rmsd_dataset, variables)


def rmsd_per_start_date(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
    area_weighted: bool = True,
) -> dict[numpy.datetime64, pandas.DataFrame]:
    """Return one pretty RMSD dataframe per forecast start date.

    The mean over the returned start-date frames reproduces ``rmsd`` exactly:
    the only difference is that the average over ``first_day_datetime`` is not
    yet applied. Emitted so the v2 runner can write per-start records.
    """
    prepared_challenger_dataset = _select_variables(_harmonise_dataset(challenger_dataset), variables)
    prepared_reference_dataset = _select_variables(_harmonise_dataset(reference_dataset), variables)
    per_start_dataset = _root_mean_squared_error_per_start(
        prepared_challenger_dataset, prepared_reference_dataset, area_weighted
    ).compute()
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    return {
        first_day_value: _to_pretty_dataframe(per_start_dataset.sel({first_day_key: first_day_value}), variables)
        for first_day_value in per_start_dataset[first_day_key].values
    }
