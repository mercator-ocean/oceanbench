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

SPATIAL_COORDINATE_ALIGNMENT_ATOL = 1e-4
SPATIAL_GRID_MINIMUM_MATCH_RATIO = 0.999
SPATIAL_COORDINATE_NAMES = (Dimension.LATITUDE.key(), Dimension.LONGITUDE.key())


def _assign_depth_dimension(dataset: xarray.Dataset) -> xarray.Dataset:
    return dataset.assign({Dimension.DEPTH.key(): [DEPTH_LABELS[depth_level] for depth_level in DepthLevel]})


def _spatial_area_weights(dataset: xarray.Dataset) -> xarray.DataArray:
    return numpy.cos(numpy.deg2rad(dataset[Dimension.LATITUDE.key()]))


def _nearest_reference_coordinate_indexes(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    coordinate_name: str,
) -> tuple[numpy.ndarray, numpy.ndarray] | None:
    challenger_coordinate = challenger_dataset[coordinate_name]
    reference_coordinate = reference_dataset[coordinate_name]

    challenger_coordinate_values = challenger_coordinate.values
    reference_coordinate_values = reference_coordinate.values

    reference_index = pandas.Index(reference_coordinate_values)
    try:
        reference_indexes = reference_index.get_indexer(
            challenger_coordinate_values,
            method="nearest",
            tolerance=SPATIAL_COORDINATE_ALIGNMENT_ATOL,
        )
    except ValueError:
        return None

    challenger_indexes = numpy.flatnonzero(reference_indexes >= 0)
    reference_indexes = reference_indexes[challenger_indexes]

    if numpy.unique(reference_indexes).size != reference_indexes.size:
        return None

    return challenger_indexes, reference_indexes


def _snap_reference_spatial_coordinates_to_challenger(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
) -> xarray.Dataset:
    reference_indexes_by_coordinate = {}
    challenger_coordinates = {}
    matched_grid_ratio = 1.0

    for coordinate_name in SPATIAL_COORDINATE_NAMES:
        coordinate_indexes = _nearest_reference_coordinate_indexes(
            challenger_dataset,
            reference_dataset,
            coordinate_name,
        )
        if coordinate_indexes is None:
            return reference_dataset

        challenger_indexes, reference_indexes = coordinate_indexes
        reference_indexes_by_coordinate[coordinate_name] = reference_indexes
        matched_grid_ratio *= challenger_indexes.size / challenger_dataset.sizes[coordinate_name]
        challenger_coordinates[coordinate_name] = challenger_dataset[coordinate_name].isel(
            {coordinate_name: challenger_indexes}
        )

    if matched_grid_ratio < SPATIAL_GRID_MINIMUM_MATCH_RATIO:
        return reference_dataset

    return reference_dataset.isel(reference_indexes_by_coordinate).assign_coords(challenger_coordinates)


def _rmsd(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
) -> xarray.Dataset:
    reference_dataset = _snap_reference_spatial_coordinates_to_challenger(challenger_dataset, reference_dataset)
    squared_error = (challenger_dataset - reference_dataset) ** 2
    area_weighted_mean_squared_error = squared_error.weighted(_spatial_area_weights(squared_error)).mean(
        dim=[Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()]
    )
    return numpy.sqrt(area_weighted_mean_squared_error).mean(dim=Dimension.FIRST_DAY_DATETIME.key())


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
        {Dimension.DEPTH.key(): [depth_level.value for depth_level in DepthLevel]},
        method="nearest",
    )
    dataset_with_depth_labels = _assign_depth_dimension(dataset_with_depth_selected)
    return dataset_with_depth_labels


def _select_variables(dataset: xarray.Dataset, variables: list[Variable]) -> xarray.Dataset:
    return dataset[[variable.key() for variable in variables]]


def rmsd(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
) -> pandas.DataFrame:
    prepared_challenger_dataset = _select_variables(_harmonise_dataset(challenger_dataset), variables)
    prepared_reference_dataset = _select_variables(_harmonise_dataset(reference_dataset), variables)
    rmsd_dataset = _rmsd(prepared_challenger_dataset, prepared_reference_dataset)
    computed_rmsd_dataset = rmsd_dataset.compute()
    return _to_pretty_dataframe(computed_rmsd_dataset, variables)
