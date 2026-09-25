# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from typing import NamedTuple

import numpy
import pandas
import xarray
from scipy import ndimage, sparse
from scipy.sparse import csgraph

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import (
    DEPTH_BINS_DEFAULT,
    DEPTH_BIN_DISPLAY_ORDER,
    Dimension,
    VARIABLE_DISPLAY_ORDER,
    VARIABLE_METADATA,
    MISSING_COUNT_COLUMN,
    SPATIAL_COORDINATE_ALIGNMENT_ATOL,
    Variable,
    is_global_longitude_grid,
)
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.ocean_mask import OCEAN_MASK_STANDARD_DEPTHS
from oceanbench.core.remote_http import with_remote_http_retries
from oceanbench.core.references.observations import load_mean_dynamic_topography
from oceanbench.core.resolution import get_dataset_resolution
from oceanbench.core.runtime_configuration import current_runtime_configuration

REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT = -0.1148
VELOCITY_TARGET_DEPTH_METERS = 15.0
OBSERVATION_COUNT_COLUMN = "Observations"
_CLASS4_OBSERVATIONS_CACHE: dict[tuple[int, int], tuple[pandas.DataFrame, numpy.ndarray, str]] = {}

# A cell is shallow when it is wet at the surface and dry at 92 metres.
CLASS4_SHALLOW_DEPTH = OCEAN_MASK_STANDARD_DEPTHS[2]
CLASS4_SHALLOW_REGION_MINIMUM_AREA_SQUARE_KILOMETERS = 100_000.0
EARTH_RADIUS_KILOMETERS = 6371.0

# The quarter degree model grid, centred on every third point of the twelfth of a degree mask.
CLASS4_COARSE_GRID_FACTOR = 3
CLASS4_COARSE_GRID_STEP_DEGREES = 0.25


class _Class4PopulationLayers(NamedTuple):
    depths: numpy.ndarray
    latitudes: numpy.ndarray
    longitudes: numpy.ndarray
    is_shallow: numpy.ndarray
    is_in_large_shallow_region: numpy.ndarray
    coarse_cells_are_wet: numpy.ndarray


# The layers are derived once per mask object; a DataArray cannot key an lru_cache.
_CLASS4_POPULATION_LAYERS_CACHE: dict[int, tuple[xarray.DataArray, _Class4PopulationLayers]] = {}


def _compute_with_remote_retries(operation_name: str, data):
    return with_remote_http_retries(operation_name, data.compute)


def _assign_depth_bins(
    depth_values: numpy.ndarray,
    depth_bins: dict[str, tuple[float, float]],
) -> numpy.ndarray:
    bin_assignments = numpy.full(len(depth_values), "", dtype=object)
    for bin_name, (depth_minimum, depth_maximum) in depth_bins.items():
        mask = (depth_values >= depth_minimum) & (depth_values < depth_maximum)
        bin_assignments[mask] = bin_name
    return bin_assignments


def _assign_temperature_depth_bins(depth_values: numpy.ndarray) -> numpy.ndarray:
    bin_assignments = _assign_depth_bins(depth_values, DEPTH_BINS_DEFAULT)
    surface_mask = (depth_values >= -1) & (depth_values < 1)
    bin_assignments[surface_mask] = "surface"
    return bin_assignments


def _interpolated_observation_record_at_target_depth(
    group_key,
    group: pandas.DataFrame,
    group_keys: list[str],
    depth_key: str,
    target_depth: float,
) -> dict[str, object] | None:
    depths = group[depth_key].to_numpy()
    values = group["observation_value"].to_numpy()
    below_depths = depths[depths <= target_depth]
    above_depths = depths[depths >= target_depth]
    if below_depths.size == 0 or above_depths.size == 0:
        return None
    below_depth = below_depths.max()
    above_depth = above_depths.min()
    below_value = values[depths == below_depth].mean()
    above_value = values[depths == above_depth].mean()
    interpolated_value = (
        below_value
        if numpy.isclose(below_depth, above_depth)
        else below_value + ((target_depth - below_depth) / (above_depth - below_depth)) * (above_value - below_value)
    )
    normalized_group_key = group_key if isinstance(group_key, tuple) else (group_key,)
    return {
        **dict(zip(group_keys, normalized_group_key)),
        "observation_value": interpolated_value,
        depth_key: target_depth,
    }


def _interpolate_observations_to_target_depth(
    observations_dataframe: pandas.DataFrame,
    target_depth: float,
) -> pandas.DataFrame:
    if observations_dataframe.empty:
        return observations_dataframe
    filtered_observations_dataframe = observations_dataframe.dropna(subset=["observation_value", Dimension.DEPTH.key()])
    if filtered_observations_dataframe.empty:
        return filtered_observations_dataframe
    time_key = Dimension.TIME.key()
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    depth_key = Dimension.DEPTH.key()
    group_keys = [time_key, latitude_key, longitude_key, "first_day", "lead_day"]
    target_columns = [
        "observation_value",
        time_key,
        latitude_key,
        longitude_key,
        "first_day",
        depth_key,
        "lead_day",
    ]

    records = [
        record
        for record in (
            _interpolated_observation_record_at_target_depth(group_key, group, group_keys, depth_key, target_depth)
            for group_key, group in filtered_observations_dataframe.groupby(group_keys, sort=False)
        )
        if record is not None
    ]

    if not records:
        return pandas.DataFrame(columns=target_columns)
    result = pandas.DataFrame.from_records(records)
    return result[target_columns]


def _prepared_class4_observations(
    observations_dataset: xarray.Dataset,
    lead_days_count: int,
) -> tuple[pandas.DataFrame, numpy.ndarray, str]:
    cache_key = (id(observations_dataset), lead_days_count)
    cached_context = _CLASS4_OBSERVATIONS_CACHE.get(cache_key)
    if cached_context is not None:
        return cached_context
    time_key = Dimension.TIME.key()
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    depth_key = Dimension.DEPTH.key()
    observation_dimension_key = observations_dataset[time_key].dims[0]

    base_subset = observations_dataset[[time_key, latitude_key, longitude_key, first_day_key, depth_key]].rename(
        {first_day_key: "first_day"}
    )
    lead_day = ((base_subset[time_key] - base_subset["first_day"]) / numpy.timedelta64(1, "D")).astype("int64")
    base_subset = base_subset.assign(lead_day=lead_day)
    valid_observation_mask = _compute_with_remote_retries(
        "Class IV observation lead-day mask read",
        (base_subset["lead_day"] >= 0) & (base_subset["lead_day"] < lead_days_count),
    )
    selected_observation_indices = numpy.flatnonzero(valid_observation_mask.values)
    base_subset = base_subset.isel({observation_dimension_key: selected_observation_indices})
    base_dataframe = (
        _compute_with_remote_retries(
            "Class IV observation coordinate read",
            base_subset,
        )
        .to_dataframe()
        .reset_index()
    )
    base_dataframe = base_dataframe.drop(columns=[observation_dimension_key], errors="ignore")
    base_dataframe = base_dataframe[[time_key, latitude_key, longitude_key, "first_day", depth_key, "lead_day"]]
    context = (base_dataframe, selected_observation_indices, observation_dimension_key)
    _CLASS4_OBSERVATIONS_CACHE[cache_key] = context
    return context


def _create_observations_dataframe(
    base_observations_dataframe: pandas.DataFrame,
    selected_observation_indices: numpy.ndarray,
    observation_dimension_key: str,
    observations_dataset: xarray.Dataset,
    observation_variable_key: str,
    standard_variable_key: str,
) -> pandas.DataFrame:
    time_key = Dimension.TIME.key()
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    depth_key = Dimension.DEPTH.key()
    observation_values = _compute_with_remote_retries(
        f"Class IV observation {standard_variable_key} read",
        observations_dataset[observation_variable_key].isel({observation_dimension_key: selected_observation_indices}),
    ).values
    valid_observation_mask = ~numpy.isnan(observation_values)
    observations_dataframe = base_observations_dataframe.loc[valid_observation_mask].copy()
    observations_dataframe["observation_value"] = observation_values[valid_observation_mask]
    observations_dataframe = observations_dataframe[
        ["observation_value", time_key, latitude_key, longitude_key, "first_day", depth_key, "lead_day"]
    ]

    if standard_variable_key in (
        Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
        Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
    ):
        observations_dataframe = _interpolate_observations_to_target_depth(
            observations_dataframe,
            VELOCITY_TARGET_DEPTH_METERS,
        )
        if observations_dataframe.empty:
            return observations_dataframe
        observations_dataframe["depth_bin"] = "15m"
        return observations_dataframe

    if standard_variable_key == Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key():
        observations_dataframe["depth_bin"] = "surface"
        return observations_dataframe
    if standard_variable_key == Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key():
        observations_dataframe["depth_bin"] = _assign_temperature_depth_bins(observations_dataframe[depth_key].values)
    else:
        observations_dataframe["depth_bin"] = _assign_depth_bins(
            observations_dataframe[depth_key].values,
            DEPTH_BINS_DEFAULT,
        )
    return observations_dataframe.loc[observations_dataframe["depth_bin"] != ""]


def create_class4_observations_dataframe(
    observations_dataset: xarray.Dataset,
    observation_variable_key: str,
    standard_variable_key: str,
    lead_days_count: int,
) -> pandas.DataFrame:
    base_observations_dataframe, selected_observation_indices, observation_dimension_key = (
        _prepared_class4_observations(
            observations_dataset,
            lead_days_count,
        )
    )
    return _create_observations_dataframe(
        base_observations_dataframe,
        selected_observation_indices,
        observation_dimension_key,
        observations_dataset,
        observation_variable_key,
        standard_variable_key,
    )


def _convert_forecast_ssh_to_sla(
    model_variable: xarray.DataArray,
    variable_key: str,
) -> xarray.DataArray:
    if variable_key != Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key():
        return model_variable
    model_dataset = rename_dataset_with_standard_names(model_variable.to_dataset(name=variable_key))
    model_variable = model_dataset[variable_key]
    resolution = get_dataset_resolution(model_variable.to_dataset(name="__resolution__"))
    mean_dynamic_topography = _mean_dynamic_topography_on_challenger_grid(
        load_mean_dynamic_topography(resolution),
        model_variable,
    )
    return model_variable - mean_dynamic_topography - REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT


def _mean_dynamic_topography_on_challenger_grid(
    mean_dynamic_topography: xarray.DataArray,
    model_variable: xarray.DataArray,
) -> xarray.DataArray:
    for coordinate_name in (Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()):
        challenger_values = model_variable[coordinate_name].values
        mean_dynamic_topography_values = mean_dynamic_topography[coordinate_name].values
        nearest_indexes = pandas.Index(mean_dynamic_topography_values).get_indexer(
            challenger_values, method="nearest", tolerance=SPATIAL_COORDINATE_ALIGNMENT_ATOL
        )
        is_inside = (challenger_values > mean_dynamic_topography_values.min() - SPATIAL_COORDINATE_ALIGNMENT_ATOL) & (
            challenger_values < mean_dynamic_topography_values.max() + SPATIAL_COORDINATE_ALIGNMENT_ATOL
        )
        if (nearest_indexes[is_inside] < 0).any():
            raise ValueError(
                f"Challenger {coordinate_name} coordinates do not match the mean dynamic topography grid "
                f"within tolerance {SPATIAL_COORDINATE_ALIGNMENT_ATOL}"
            )
        challenger_coordinate = {coordinate_name: model_variable[coordinate_name]}
        mean_dynamic_topography = mean_dynamic_topography.reindex(
            challenger_coordinate, method="nearest", tolerance=SPATIAL_COORDINATE_ALIGNMENT_ATOL
        ).assign_coords(challenger_coordinate)
    return mean_dynamic_topography


def prepare_class4_model_variable(
    model_variable: xarray.DataArray,
    variable_key: str,
) -> xarray.DataArray:
    return _convert_forecast_ssh_to_sla(model_variable, variable_key)


def _bracketing_level_indices(
    sorted_depths: numpy.ndarray,
    target_depths: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    insert_idx = numpy.searchsorted(sorted_depths, target_depths)
    idx_upper = numpy.clip(insert_idx, 0, len(sorted_depths) - 1)
    idx_lower = numpy.clip(insert_idx - 1, 0, len(sorted_depths) - 1)

    exact_mask = sorted_depths[idx_upper] == target_depths
    idx_lower = numpy.where(exact_mask, idx_upper, idx_lower)
    return idx_lower, idx_upper


def _interpolate_vertically_bracket(
    profiles: numpy.ndarray,
    model_depths: numpy.ndarray,
    target_depths: numpy.ndarray,
) -> numpy.ndarray:
    if len(model_depths) == 1:
        return profiles[0, :]
    observation_count = profiles.shape[1]
    result = numpy.full(observation_count, numpy.nan)
    sort_order = numpy.argsort(model_depths)
    sorted_depths = model_depths[sort_order]
    sorted_profiles = profiles[sort_order, :]

    idx_lower, idx_upper = _bracketing_level_indices(sorted_depths, target_depths)

    obs_indices = numpy.arange(observation_count)
    lower_values = sorted_profiles[idx_lower, obs_indices]
    upper_values = sorted_profiles[idx_upper, obs_indices]
    lower_depths = sorted_depths[idx_lower]
    upper_depths = sorted_depths[idx_upper]

    same_depth = numpy.isclose(lower_depths, upper_depths)
    interpolated = numpy.empty(observation_count, dtype=float)
    interpolated[same_depth] = lower_values[same_depth]
    different = ~same_depth
    if numpy.any(different):
        weights = (target_depths[different] - lower_depths[different]) / (
            upper_depths[different] - lower_depths[different]
        )
        interpolated[different] = lower_values[different] + weights * (
            upper_values[different] - lower_values[different]
        )

    bracket_is_valid = ~numpy.isnan(lower_values) & ~numpy.isnan(upper_values)
    result[bracket_is_valid] = interpolated[bracket_is_valid]
    return result


def _model_data_with_depth_dimension(model_data: xarray.DataArray) -> xarray.DataArray:
    depth_key = Dimension.DEPTH.key()
    if depth_key in model_data.dims:
        return model_data
    return model_data.expand_dims({depth_key: [0.0]})


def _linearly_interpolated_profiles(
    data: xarray.DataArray,
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
) -> numpy.ndarray:
    interpolated_profiles = data.interp(
        {
            Dimension.LATITUDE.key(): xarray.DataArray(latitudes, dims="observation"),
            Dimension.LONGITUDE.key(): xarray.DataArray(longitudes, dims="observation"),
        },
        method="linear",
    )
    return interpolated_profiles.compute().values


def _horizontally_interpolated_profiles(
    time_slice: xarray.DataArray,
    observation_group: pandas.DataFrame,
) -> numpy.ndarray:
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    observation_latitudes = observation_group[latitude_key].values
    observation_longitudes = observation_group[longitude_key].values
    grid_longitudes = time_slice[longitude_key].values
    first_longitude, last_longitude = grid_longitudes[0], grid_longitudes[-1]
    is_on_grid = (observation_longitudes >= first_longitude) & (observation_longitudes <= last_longitude)
    if not is_global_longitude_grid(grid_longitudes) or is_on_grid.all():
        return _linearly_interpolated_profiles(time_slice, observation_latitudes, observation_longitudes)

    wrapped_longitudes = numpy.where(
        is_on_grid,
        observation_longitudes,
        first_longitude + (observation_longitudes - first_longitude) % 360,
    )
    is_in_seam = wrapped_longitudes > last_longitude
    seam_columns = time_slice.isel({longitude_key: [-1, 0]}).assign_coords(
        {longitude_key: [last_longitude, first_longitude + 360]}
    )
    profile_shape = [
        time_slice.sizes[dimension] for dimension in time_slice.dims if dimension not in (latitude_key, longitude_key)
    ]
    interpolated_profiles = numpy.full(profile_shape + [len(observation_group)], numpy.nan)
    for data, is_selected in ((time_slice, ~is_in_seam), (seam_columns, is_in_seam)):
        if is_selected.any():
            interpolated_profiles[..., is_selected] = _linearly_interpolated_profiles(
                data,
                observation_latitudes[is_selected],
                wrapped_longitudes[is_selected],
            )
    return interpolated_profiles


def _interpolated_model_values_for_observation_group(
    time_slice: xarray.DataArray,
    observation_group: pandas.DataFrame,
    model_depths: numpy.ndarray,
) -> numpy.ndarray:
    observation_depths = observation_group[Dimension.DEPTH.key()].values
    horizontally_interpolated = _horizontally_interpolated_profiles(time_slice, observation_group)
    return _interpolate_vertically_bracket(
        horizontally_interpolated,
        model_depths,
        observation_depths,
    )


def _assign_model_values_for_first_day(
    model_values: numpy.ndarray,
    model_data: xarray.DataArray,
    first_day_group: pandas.DataFrame,
    first_day_index: int,
    lead_day_to_index: dict[object, int],
    model_depths: numpy.ndarray,
    variable_key: str,
) -> None:
    first_day_block = (
        model_data.isel({Dimension.FIRST_DAY_DATETIME.key(): first_day_index}).compute()
        if current_runtime_configuration().class4_fast_interpolation
        else None
    )
    for lead_day, observation_group in first_day_group.groupby("lead_day", sort=False):
        time_slice = (
            first_day_block.isel({Dimension.LEAD_DAY_INDEX.key(): lead_day_to_index[lead_day]})
            if first_day_block is not None
            else _compute_with_remote_retries(
                f"Class IV model {variable_key} read for lead day {lead_day}",
                model_data.isel(
                    {
                        Dimension.FIRST_DAY_DATETIME.key(): first_day_index,
                        Dimension.LEAD_DAY_INDEX.key(): lead_day_to_index[lead_day],
                    }
                ),
            )
        )
        model_values[observation_group.index.values] = _interpolated_model_values_for_observation_group(
            time_slice,
            observation_group,
            model_depths,
        )


def _interpolate_model_to_observations(
    model_data: xarray.DataArray,
    observations_dataframe: pandas.DataFrame,
    variable_key: str,
) -> numpy.ndarray:
    observations_dataframe = observations_dataframe.reset_index(drop=True)
    model_data = _model_data_with_depth_dimension(model_data)
    model_depths = model_data[Dimension.DEPTH.key()].values
    first_days = model_data[Dimension.FIRST_DAY_DATETIME.key()].values
    lead_days = model_data[Dimension.LEAD_DAY_INDEX.key()].values
    first_day_to_index = {first_day: index for index, first_day in enumerate(first_days)}
    lead_day_to_index = {lead_day: index for index, lead_day in enumerate(lead_days)}
    model_values = numpy.full(len(observations_dataframe), numpy.nan)
    for first_day, first_day_group in observations_dataframe.groupby("first_day", sort=False):
        _assign_model_values_for_first_day(
            model_values,
            model_data,
            first_day_group,
            first_day_to_index[first_day],
            lead_day_to_index,
            model_depths,
            variable_key,
        )
    return model_values


def interpolate_class4_model_to_observations(
    model_data: xarray.DataArray,
    observations_dataframe: pandas.DataFrame,
) -> numpy.ndarray:
    variable_key = str(model_data.name)
    return _interpolate_model_to_observations(model_data, observations_dataframe, variable_key)


def _dateline_neighbour_labels(labels: numpy.ndarray, row_shift: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    rows = numpy.arange(max(0, -row_shift), min(labels.shape[0], labels.shape[0] - row_shift))
    return labels[rows, 0], labels[rows + row_shift, -1]


def _labels_linked_across_the_dateline(labels: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    neighbour_labels = [_dateline_neighbour_labels(labels, row_shift) for row_shift in (-1, 0, 1)]
    first_labels = numpy.concatenate([first for first, _ in neighbour_labels])
    last_labels = numpy.concatenate([last for _, last in neighbour_labels])
    is_linked = (first_labels > 0) & (last_labels > 0)
    return first_labels[is_linked], last_labels[is_linked]


def _is_in_large_shallow_region(
    is_shallow: numpy.ndarray,
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
) -> numpy.ndarray:
    labels, label_count = ndimage.label(is_shallow, structure=numpy.ones((3, 3), dtype=int))
    first_labels, last_labels = _labels_linked_across_the_dateline(labels)
    label_graph = sparse.coo_matrix(
        (numpy.ones(len(first_labels)), (first_labels, last_labels)),
        shape=(label_count + 1, label_count + 1),
    )
    _, region_of_label = csgraph.connected_components(label_graph, directed=False)
    region_of_cell = region_of_label[labels]
    cell_area = (
        (EARTH_RADIUS_KILOMETERS**2)
        * numpy.deg2rad(abs(latitudes[1] - latitudes[0]))
        * numpy.deg2rad(abs(longitudes[1] - longitudes[0]))
        * numpy.cos(numpy.deg2rad(latitudes))
    )
    region_area = numpy.bincount(
        region_of_cell.ravel(),
        weights=numpy.broadcast_to(cell_area[:, numpy.newaxis], is_shallow.shape).ravel(),
    )
    region_area[region_of_label[0]] = 0.0
    return region_area[region_of_cell] > CLASS4_SHALLOW_REGION_MINIMUM_AREA_SQUARE_KILOMETERS


def _coarse_cells_are_wet(is_wet: numpy.ndarray) -> numpy.ndarray:
    _, latitude_count, longitude_count = is_wet.shape
    coarse_rows = numpy.arange(0, latitude_count, CLASS4_COARSE_GRID_FACTOR)
    coarse_columns = numpy.arange(0, longitude_count, CLASS4_COARSE_GRID_FACTOR)
    return numpy.logical_and.reduce(
        [
            is_wet[:, numpy.clip(coarse_rows + row_offset, 0, latitude_count - 1)][
                :, :, numpy.mod(coarse_columns + column_offset, longitude_count)
            ]
            for row_offset in (-1, 0, 1)
            for column_offset in (-1, 0, 1)
        ]
    )


def _class4_population_layers(ocean_mask: xarray.DataArray) -> _Class4PopulationLayers:
    cached_layers = _CLASS4_POPULATION_LAYERS_CACHE.get(id(ocean_mask))
    if cached_layers is not None and cached_layers[0] is ocean_mask:
        return cached_layers[1]
    sorted_mask = ocean_mask.sortby(Dimension.DEPTH.key())
    latitudes = sorted_mask[Dimension.LATITUDE.key()].values.astype("float64")
    longitudes = sorted_mask[Dimension.LONGITUDE.key()].values.astype("float64")
    is_wet = sorted_mask.values.astype(bool)
    is_shallow = is_wet[0] & ~sorted_mask.sel(
        {Dimension.DEPTH.key(): CLASS4_SHALLOW_DEPTH}, method="nearest"
    ).values.astype(bool)
    layers = _Class4PopulationLayers(
        depths=sorted_mask[Dimension.DEPTH.key()].values,
        latitudes=latitudes,
        longitudes=longitudes,
        is_shallow=is_shallow,
        is_in_large_shallow_region=_is_in_large_shallow_region(is_shallow, latitudes, longitudes),
        coarse_cells_are_wet=_coarse_cells_are_wet(is_wet),
    )
    _CLASS4_POPULATION_LAYERS_CACHE.clear()
    _CLASS4_POPULATION_LAYERS_CACHE[id(ocean_mask)] = (ocean_mask, layers)
    return layers


def _surrounding_cells(
    row_below: numpy.ndarray,
    column_left: numpy.ndarray,
    row_count: int,
    column_count: int,
) -> list[tuple[numpy.ndarray, numpy.ndarray]]:
    first_row = numpy.clip(row_below, 0, row_count - 1)
    rows = [first_row, numpy.clip(first_row + 1, 0, row_count - 1)]
    columns = [numpy.mod(column_left, column_count), numpy.mod(column_left + 1, column_count)]
    return [(row, column) for row in rows for column in columns]


def class4_observations_in_shared_population(
    observations_dataframe: pandas.DataFrame,
    ocean_mask: xarray.DataArray,
) -> pandas.DataFrame:
    """
    Keep only the observations of the OceanBench Class IV population, built from the ocean mask alone.

    An observation is dropped when one of its four surrounding twelfth of a degree cells is shallow,
    wet at the surface but dry at 92 metres, unless one of those shallow cells belongs to a shallow
    region larger than 100,000 square kilometres. It is also dropped unless its four surrounding
    quarter degree cells, each wet only when its nine twelfth of a degree cells are, are wet at its
    deeper bracketing mask depth. The population is therefore the same for every challenger and
    every reference, whatever their grids.
    """
    observations_dataframe = observations_dataframe.reset_index(drop=True)
    layers = _class4_population_layers(ocean_mask)
    latitudes = observations_dataframe[Dimension.LATITUDE.key()].values
    longitudes = observations_dataframe[Dimension.LONGITUDE.key()].values

    fine_cells = _surrounding_cells(
        numpy.searchsorted(layers.latitudes, latitudes, side="right") - 1,
        numpy.searchsorted(layers.longitudes, longitudes, side="right") - 1,
        len(layers.latitudes),
        len(layers.longitudes),
    )
    has_shallow_cell = numpy.logical_or.reduce([layers.is_shallow[cell] for cell in fine_cells])
    has_large_shallow_region_cell = numpy.logical_or.reduce(
        [layers.is_in_large_shallow_region[cell] for cell in fine_cells]
    )

    _, coarse_row_count, coarse_column_count = layers.coarse_cells_are_wet.shape
    coarse_cells = _surrounding_cells(
        numpy.floor((latitudes - layers.latitudes[0]) / CLASS4_COARSE_GRID_STEP_DEGREES).astype(numpy.int64),
        numpy.floor((longitudes - layers.longitudes[0]) / CLASS4_COARSE_GRID_STEP_DEGREES).astype(numpy.int64),
        coarse_row_count,
        coarse_column_count,
    )
    _, deeper_level = _bracketing_level_indices(
        layers.depths,
        observations_dataframe[Dimension.DEPTH.key()].values,
    )
    has_wet_coarse_cells = numpy.logical_and.reduce(
        [layers.coarse_cells_are_wet[deeper_level, row, column] for row, column in coarse_cells]
    )

    is_eligible = (~has_shallow_cell | has_large_shallow_region_cell) & has_wet_coarse_cells
    return observations_dataframe.loc[is_eligible]


def _compute_rmsd_table(
    dataframe: pandas.DataFrame,
    variable_key: str,
) -> pandas.DataFrame:
    eligible_dataframe = dataframe.dropna(subset=["observation_value"])
    grouped = (
        eligible_dataframe.assign(
            squared_difference=(eligible_dataframe["model_value"] - eligible_dataframe["observation_value"]) ** 2,
            missing=eligible_dataframe["model_value"].isna(),
        )
        .groupby(["depth_bin", "lead_day"], as_index=False)
        .agg(
            rmsd=("squared_difference", lambda values: numpy.sqrt(values.mean())),
            count=("squared_difference", "size"),
            missing=("missing", "sum"),
        )
    )
    grouped["count"] = grouped["count"].astype(int)
    grouped["missing"] = grouped["missing"].astype(int)
    grouped["variable"] = variable_key
    return grouped[["variable", "depth_bin", "lead_day", "rmsd", "count", "missing"]]


def compute_class4_rmsd_table(
    dataframe: pandas.DataFrame,
    variable_key: str,
) -> pandas.DataFrame:
    return _compute_rmsd_table(dataframe, variable_key)


def _observation_variable_depth_label(standard_name: str, depth_bin: str) -> str:
    display_name, unit = VARIABLE_METADATA[standard_name]
    if standard_name == Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key():
        # TODO: replace the reused SSH standard name with an agreed SLA-specific one for Class IV observations.
        display_name = "sea level anomaly"
    return f"{display_name.capitalize()} ({unit}) [{standard_name}]{{{depth_bin}}}"


def format_class4_results(results_dataframe: pandas.DataFrame, lead_days_count: int) -> pandas.DataFrame:
    scored_pairs = pandas.MultiIndex.from_frame(results_dataframe[["variable", "depth_bin"]].drop_duplicates())
    pivot_table = (
        results_dataframe.pivot_table(
            values="rmsd",
            index=["variable", "depth_bin"],
            columns="lead_day",
            aggfunc="first",
        )
        .reindex(index=scored_pairs, columns=range(lead_days_count))
        .reset_index()
    )
    first_available_day = results_dataframe["lead_day"].min()
    observation_counts = results_dataframe[results_dataframe["lead_day"] == first_available_day][
        ["variable", "depth_bin", "count", "missing"]
    ]
    pivot_table = pivot_table.merge(observation_counts, on=["variable", "depth_bin"], how="left")
    pivot_table["variable_sort"] = pivot_table["variable"].map(VARIABLE_DISPLAY_ORDER).astype(float)
    pivot_table["depth_sort"] = pivot_table["depth_bin"].map(DEPTH_BIN_DISPLAY_ORDER)
    pivot_table = pivot_table.sort_values(["variable_sort", "depth_sort"]).drop(columns=["variable_sort", "depth_sort"])
    pivot_table["label"] = pivot_table.apply(
        lambda row: _observation_variable_depth_label(row["variable"], row["depth_bin"]),
        axis=1,
    )

    lead_columns = [column for column in pivot_table.columns if isinstance(column, (int, numpy.integer))]
    lead_labels = lead_day_labels(1, lead_days_count)
    column_rename = {column: lead_labels[column] for column in lead_columns}
    result = pivot_table.set_index("label")[lead_columns + ["count", "missing"]].rename(
        columns=column_rename | {"count": OBSERVATION_COUNT_COLUMN, "missing": MISSING_COUNT_COLUMN}
    )
    result.index.name = None
    result.columns.name = None
    return result


def class4_variable_results(
    challenger: xarray.Dataset,
    observations: xarray.Dataset,
    base_observations_dataframe: pandas.DataFrame,
    selected_observation_indices: numpy.ndarray,
    observation_dimension_key: str,
    observation_variable_key: str,
    challenger_variable_key: str,
    standard_variable_key: str,
) -> pandas.DataFrame:
    observations_dataframe = _create_observations_dataframe(
        base_observations_dataframe,
        selected_observation_indices,
        observation_dimension_key,
        observations,
        observation_variable_key,
        standard_variable_key,
    )
    if observations_dataframe.empty:
        return pandas.DataFrame()

    observations_dataframe = observations_dataframe.dropna(subset=["observation_value"])
    model_variable = _convert_forecast_ssh_to_sla(
        challenger[challenger_variable_key],
        standard_variable_key,
    )
    observations_dataframe = observations_dataframe.assign(
        model_value=_interpolate_model_to_observations(
            model_variable,
            observations_dataframe,
            standard_variable_key,
        )
    )
    return _compute_rmsd_table(observations_dataframe, standard_variable_key)
