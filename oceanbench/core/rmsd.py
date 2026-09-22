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

MISSING_COUNT_COLUMN = "Missing"
MISSING_FRACTION_COLUMN = "Missing fraction"

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
) -> tuple[numpy.ndarray, numpy.ndarray]:
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
    except (ValueError, pandas.errors.InvalidIndexError) as error:
        raise ValueError(
            f"Could not align {coordinate_name} coordinates: nearest-neighbor lookup failed: {error}"
        ) from error

    challenger_indexes = numpy.flatnonzero(reference_indexes >= 0)
    reference_indexes = reference_indexes[challenger_indexes]

    if numpy.unique(reference_indexes).size != reference_indexes.size:
        raise ValueError(
            f"Could not align {coordinate_name} coordinates: multiple challenger coordinates match the same "
            f"reference coordinate within tolerance {SPATIAL_COORDINATE_ALIGNMENT_ATOL}"
        )

    return challenger_indexes, reference_indexes


def _snap_reference_spatial_coordinates_to_challenger(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
) -> xarray.Dataset:
    reference_indexes_by_coordinate = {}
    challenger_coordinates = {}
    coordinate_match_ratios_by_name = {}
    matched_grid_ratio = 1.0

    for coordinate_name in SPATIAL_COORDINATE_NAMES:
        coordinate_indexes = _nearest_reference_coordinate_indexes(
            challenger_dataset,
            reference_dataset,
            coordinate_name,
        )

        challenger_indexes, reference_indexes = coordinate_indexes
        reference_indexes_by_coordinate[coordinate_name] = reference_indexes
        coordinate_match_ratio = challenger_indexes.size / challenger_dataset.sizes[coordinate_name]
        coordinate_match_ratios_by_name[coordinate_name] = coordinate_match_ratio
        matched_grid_ratio *= coordinate_match_ratio
        challenger_coordinates[coordinate_name] = challenger_dataset[coordinate_name].isel(
            {coordinate_name: challenger_indexes}
        )

    if matched_grid_ratio < SPATIAL_GRID_MINIMUM_MATCH_RATIO:
        coordinate_match_ratios = ", ".join(
            f"{coordinate_name}={coordinate_match_ratios_by_name[coordinate_name]:.4%}"
            for coordinate_name in SPATIAL_COORDINATE_NAMES
        )
        raise ValueError(
            "Could not align reference spatial grid to challenger spatial grid: "
            f"matched {matched_grid_ratio:.4%} of challenger grid points, "
            f"required at least {SPATIAL_GRID_MINIMUM_MATCH_RATIO:.4%}; "
            f"coordinate match ratios: {coordinate_match_ratios}; "
            f"tolerance={SPATIAL_COORDINATE_ALIGNMENT_ATOL}"
        )

    return reference_dataset.isel(reference_indexes_by_coordinate).assign_coords(challenger_coordinates)


def _ocean_mask_on_challenger_grid(
    ocean_mask: xarray.DataArray,
    challenger_dataset: xarray.Dataset,
) -> xarray.DataArray:
    """
    Put the OceanBench ocean mask on the challenger grid, by nearest neighbour.

    Nearest neighbour rather than a conservative remapping: on a coarser challenger grid a cell is
    wet when the twelfth of a degree cell at its centre is wet, which is simple to state and to
    reproduce, at the price of ignoring the sub-cell coastline.
    """
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    regridded_mask = ocean_mask.sel(
        {
            latitude_key: challenger_dataset[latitude_key],
            longitude_key: challenger_dataset[longitude_key],
        },
        method="nearest",
    )
    return regridded_mask.assign_coords(
        {
            Dimension.DEPTH.key(): [DEPTH_LABELS[level] for level in DepthLevel],
            latitude_key: challenger_dataset[latitude_key],
            longitude_key: challenger_dataset[longitude_key],
        }
    )


def _variable_ocean_mask(
    ocean_mask: xarray.DataArray,
    dataset: xarray.Dataset,
    variable_name: str,
) -> xarray.DataArray:
    if Dimension.DEPTH.key() in dataset[variable_name].dims:
        return ocean_mask
    surface_mask = ocean_mask.sel({Dimension.DEPTH.key(): DEPTH_LABELS[DepthLevel.SURFACE]})
    return surface_mask.drop_vars(Dimension.DEPTH.key())


def _masked_to_ocean(dataset: xarray.Dataset, ocean_mask: xarray.DataArray) -> xarray.Dataset:
    return dataset.assign(
        {
            variable_name: dataset[variable_name].where(_variable_ocean_mask(ocean_mask, dataset, variable_name))
            for variable_name in dataset.data_vars
        }
    )


def _missing_fraction_key(variable_name: str) -> str:
    return f"{variable_name}_missing_fraction"


def _missing_ocean_cells(
    challenger_dataset: xarray.Dataset,
    ocean_mask: xarray.DataArray,
    variable_name: str,
) -> tuple[xarray.DataArray, xarray.DataArray]:
    spatial_dimensions = [Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()]
    forecast_dimensions = [Dimension.FIRST_DAY_DATETIME.key(), Dimension.LEAD_DAY_INDEX.key()]
    variable_mask = _variable_ocean_mask(ocean_mask, challenger_dataset, variable_name)
    is_missing = variable_mask & challenger_dataset[variable_name].isnull()
    missing_count = is_missing.sum(dim=spatial_dimensions).mean(dim=forecast_dimensions)
    missing_fraction = (
        is_missing.where(variable_mask)
        .weighted(_spatial_area_weights(challenger_dataset))
        .mean(dim=spatial_dimensions)
        .mean(dim=forecast_dimensions)
    )
    return missing_count, missing_fraction


def _missing_counts(challenger_dataset: xarray.Dataset, ocean_mask: xarray.DataArray) -> xarray.Dataset:
    """
    Count, per variable and depth, the ocean cells the challenger leaves empty.

    The count and the area weighted fraction are averaged over initialization days and lead days,
    so they describe the challenger rather than one particular forecast.
    """
    missing_by_variable = {
        variable_name: _missing_ocean_cells(challenger_dataset, ocean_mask, variable_name)
        for variable_name in challenger_dataset.data_vars
    }
    missing_counts = {variable_name: count for variable_name, (count, _) in missing_by_variable.items()}
    missing_fractions = {
        _missing_fraction_key(variable_name): fraction for variable_name, (_, fraction) in missing_by_variable.items()
    }
    return xarray.Dataset(missing_counts | missing_fractions)


def _rmsd(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    ocean_mask: xarray.DataArray | None = None,
) -> xarray.Dataset:
    reference_dataset = _snap_reference_spatial_coordinates_to_challenger(challenger_dataset, reference_dataset)
    if ocean_mask is not None:
        challenger_dataset = _masked_to_ocean(challenger_dataset, ocean_mask)
        reference_dataset = _masked_to_ocean(reference_dataset, ocean_mask)
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


def _scored_variable_depth_pairs(dataset: xarray.Dataset, variables: list[Variable]) -> list[tuple[str, str]]:
    return [
        (variable.key(), depth_level)
        for depth_level in DEPTH_LABELS.values()
        for variable in variables
        if depth_level == DEPTH_LABELS[DepthLevel.SURFACE] or _has_depths(dataset, variable.key())
    ]


def _missing_value(missing_dataset: xarray.Dataset, missing_key: str, depth_level: str) -> float:
    missing_array = missing_dataset[missing_key]
    if Dimension.DEPTH.key() in missing_array.dims:
        return float(missing_array.sel({Dimension.DEPTH.key(): depth_level}))
    return float(missing_array)


def _to_pretty_dataframe(
    dataset: xarray.Dataset,
    variables: list[Variable],
    missing_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    dataset_with_depth = _assign_depth_dimension(dataset) if dataset.get(Dimension.DEPTH.key()) is None else dataset
    scored_pairs = _scored_variable_depth_pairs(dataset_with_depth, variables)
    values_2d: dict[str, numpy.ndarray] = {
        _variable_depth_label(dataset_with_depth, variable_key, depth_level): _select_dataset_variable_and_depth(
            dataset_with_depth, variable_key, depth_level
        )
        for variable_key, depth_level in scored_pairs
    }
    lead_days_count = dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    pretty_dataframe = pandas.DataFrame(values_2d).set_index([lead_day_labels(1, lead_days_count)]).T
    pretty_dataframe[MISSING_COUNT_COLUMN] = [
        round(_missing_value(missing_dataset, variable_key, depth_level)) for variable_key, depth_level in scored_pairs
    ]
    pretty_dataframe[MISSING_FRACTION_COLUMN] = [
        _missing_value(missing_dataset, _missing_fraction_key(variable_key), depth_level)
        for variable_key, depth_level in scored_pairs
    ]
    return pretty_dataframe


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
    ocean_mask: xarray.DataArray,
) -> pandas.DataFrame:
    """
    Area weighted gridded RMSD, on the cells the OceanBench ocean mask calls ocean.

    A cell is scored when the mask is wet there and both the challenger and the reference have a
    value, the reference having first been snapped to the challenger grid by nearest index. The
    ocean cells the challenger leaves empty are not scored but are reported, as a count and as an
    area weighted fraction, in the Missing columns.
    """
    prepared_challenger_dataset = _select_variables(_harmonise_dataset(challenger_dataset), variables)
    prepared_reference_dataset = _select_variables(_harmonise_dataset(reference_dataset), variables)
    challenger_ocean_mask = _ocean_mask_on_challenger_grid(ocean_mask, prepared_challenger_dataset)
    missing_dataset = _missing_counts(prepared_challenger_dataset, challenger_ocean_mask).compute()
    rmsd_dataset = _rmsd(prepared_challenger_dataset, prepared_reference_dataset, challenger_ocean_mask)
    computed_rmsd_dataset = rmsd_dataset.compute()
    return _to_pretty_dataframe(computed_rmsd_dataset, variables, missing_dataset)
