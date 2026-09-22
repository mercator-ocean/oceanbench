# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import xarray

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import (
    DEPTH_BINS_DEFAULT,
    DEPTH_BIN_DISPLAY_ORDER,
    Dimension,
    VARIABLE_DISPLAY_ORDER,
    VARIABLE_METADATA,
    Variable,
)
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.remote_http import with_remote_http_retries
from oceanbench.core.references.observations import load_mean_dynamic_topography
from oceanbench.core.resolution import get_dataset_resolution
from oceanbench.core.runtime_configuration import current_runtime_configuration

REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT = -0.1148
VELOCITY_TARGET_DEPTH_METERS = 15.0
OBSERVATION_COUNT_COLUMN = "Observations"
MISSING_COUNT_COLUMN = "Missing"
_CLASS4_OBSERVATIONS_CACHE: dict[tuple[int, int], tuple[pandas.DataFrame, numpy.ndarray, str]] = {}


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
    mean_dynamic_topography = load_mean_dynamic_topography(resolution)
    return model_variable - mean_dynamic_topography - REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT


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


def _horizontally_interpolated_profiles(
    time_slice: xarray.DataArray,
    observation_group: pandas.DataFrame,
) -> numpy.ndarray:
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    observation_latitudes = observation_group[latitude_key].values
    observation_longitudes = observation_group[longitude_key].values
    interpolated_profiles = time_slice.interp(
        {
            latitude_key: xarray.DataArray(observation_latitudes, dims="observation"),
            longitude_key: xarray.DataArray(observation_longitudes, dims="observation"),
        },
        method="linear",
    )
    return interpolated_profiles.compute().values


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


def gate_class4_observations_to_reference_population(
    observations_dataframe: pandas.DataFrame,
    ocean_mask: xarray.DataArray,
) -> pandas.DataFrame:
    """
    Keep only the observations the OceanBench ocean mask brackets on the standard depth grid.

    The mask is carried as one where OceanBench considers the cell ocean and not a number where it
    does not, and it goes through the same horizontal linear interpolation as a challenger, so an
    observation is over the ocean exactly when that interpolation stays finite. It is kept only
    when both of its bracketing standard depths are finite there, so the scored population depends
    neither on the challenger vertical axis nor on the reference a metric happens to use.
    """
    observations_dataframe = observations_dataframe.reset_index(drop=True)
    finite_mask = ocean_mask.where(ocean_mask, numpy.nan).astype(float)
    wet_profiles = _horizontally_interpolated_profiles(
        finite_mask,
        observations_dataframe,
    )
    mask_depths = ocean_mask[Dimension.DEPTH.key()].values
    sort_order = numpy.argsort(mask_depths)
    sorted_depths = mask_depths[sort_order]
    sorted_wet_profiles = wet_profiles[sort_order, :]

    idx_lower, idx_upper = _bracketing_level_indices(
        sorted_depths,
        observations_dataframe[Dimension.DEPTH.key()].values,
    )
    obs_indices = numpy.arange(len(observations_dataframe))
    is_eligible = numpy.isfinite(sorted_wet_profiles[idx_lower, obs_indices]) & numpy.isfinite(
        sorted_wet_profiles[idx_upper, obs_indices]
    )
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
    pivot_table = results_dataframe.pivot_table(
        values="rmsd",
        index=["variable", "depth_bin"],
        columns="lead_day",
        aggfunc="first",
    ).reset_index()
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
