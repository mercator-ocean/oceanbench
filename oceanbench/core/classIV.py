# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray
import pandas
from scipy.interpolate import CubicSpline
from pathlib import Path
import shutil

from oceanbench.core.dataset_utils import (
    Dimension,
    Variable,
    VARIABLE_METADATA,
    DEPTH_BINS_DEFAULT,
    VARIABLE_DISPLAY_ORDER,
    DEPTH_BIN_DISPLAY_ORDER,
)
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.local_stage import local_stage_build_guard, local_stage_directory, should_stage_locally
from oceanbench.core.climate_forecast_standard_names import (
    rename_dataset_with_standard_names,
)
from oceanbench.core.resolution import get_dataset_resolution
from oceanbench.core.references.observations import LOCAL_STAGE_OBSERVATIONS_KEY

REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT = -0.1148
MINIMUM_POINTS_FOR_CUBIC_SPLINE = 4
VERTICAL_INTERPOLATION_BATCH_SIZE = 1000
VELOCITY_TARGET_DEPTH_METERS = 15.0


def _mean_dynamic_topography_zarr_url(resolution: str) -> str:
    if resolution == "twelfth_degree":
        return "https://minio.dive.edito.eu/project-oceanbench/public/glorys12_mdt_2024/" "GLO-MFC_001_030_mdt.zarr"
    if resolution == "quarter_degree":
        return (
            "https://minio.dive.edito.eu/project-oceanbench/public/glorys12_mdt_2024/" "GLO-MFC_001_030_mdt_025deg.zarr"
        )
    if resolution == "one_degree":
        return (
            "https://minio.dive.edito.eu/project-oceanbench/public/glorys12_mdt_2024/" "GLO-MFC_001_030_mdt_1_deg.zarr"
        )
    raise ValueError(f"Unsupported resolution : {resolution}.")


def _mean_dynamic_topography_stage_path(resolution: str) -> Path:
    return local_stage_directory() / f"class4-mean-dynamic-topography-2024-{resolution}.zarr"


def _write_staged_mean_dynamic_topography_dataset(
    mean_dynamic_topography_dataset: xarray.Dataset,
    stage_path: Path,
) -> None:
    temporary_stage_path = stage_path.with_name(f"{stage_path.name}.tmp")
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(temporary_stage_path, ignore_errors=True)
    mean_dynamic_topography_dataset.to_zarr(temporary_stage_path, mode="w")
    shutil.rmtree(stage_path, ignore_errors=True)
    temporary_stage_path.rename(stage_path)


def _open_mean_dynamic_topography_dataset(resolution: str) -> xarray.Dataset:
    mean_dynamic_topography_url = _mean_dynamic_topography_zarr_url(resolution)
    if not should_stage_locally(LOCAL_STAGE_OBSERVATIONS_KEY):
        return xarray.open_dataset(
            mean_dynamic_topography_url,
            engine="zarr",
            chunks="auto",
            consolidated=True,
        )
    local_stage_path = _mean_dynamic_topography_stage_path(resolution)
    with local_stage_build_guard(local_stage_path) as should_build_stage:
        if should_build_stage:
            mean_dynamic_topography_dataset = xarray.open_dataset(
                mean_dynamic_topography_url,
                engine="zarr",
                chunks="auto",
                consolidated=True,
            )
            try:
                _write_staged_mean_dynamic_topography_dataset(mean_dynamic_topography_dataset, local_stage_path)
            finally:
                mean_dynamic_topography_dataset.close()
    return xarray.open_dataset(local_stage_path, engine="zarr")


def _load_mean_dynamic_topography(resolution: str) -> xarray.DataArray:
    dataset = _open_mean_dynamic_topography_dataset(resolution)
    dataset = rename_dataset_with_standard_names(dataset)
    return dataset[Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()]


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
    observation_values = (
        observations_dataset[observation_variable_key]
        .isel({observation_dimension_key: selected_observation_indices})
        .compute()
        .values
    )
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


def _create_observations_base_dataframe(
    observations_dataset: xarray.Dataset,
    lead_days_count: int,
) -> tuple[pandas.DataFrame, numpy.ndarray, str]:
    time_key = Dimension.TIME.key()
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    depth_key = Dimension.DEPTH.key()
    observation_dimension_key = observations_dataset[time_key].dims[0]

    base_subset = observations_dataset[[time_key, latitude_key, longitude_key, first_day_key, depth_key]].rename(
        {first_day_key: "first_day"}
    )
    lead_day = ((base_subset[time_key] - base_subset["first_day"]) / numpy.timedelta64(1, "D")).astype("int64") - 1
    base_subset = base_subset.assign(lead_day=lead_day)
    valid_observation_mask = ((base_subset["lead_day"] >= 0) & (base_subset["lead_day"] < lead_days_count)).compute()
    selected_observation_indices = numpy.flatnonzero(valid_observation_mask.values)
    base_subset = base_subset.isel({observation_dimension_key: selected_observation_indices})
    base_dataframe = base_subset.compute().to_dataframe().reset_index()
    base_dataframe = base_dataframe.drop(columns=[observation_dimension_key], errors="ignore")
    base_dataframe = base_dataframe[[time_key, latitude_key, longitude_key, "first_day", depth_key, "lead_day"]]
    return base_dataframe, selected_observation_indices, observation_dimension_key


def _convert_forecast_ssh_to_sla(
    model_variable: xarray.DataArray,
    variable_key: str,
) -> xarray.DataArray:
    if variable_key != Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key():
        return model_variable
    model_dataset = rename_dataset_with_standard_names(model_variable.to_dataset(name=variable_key))
    model_variable = model_dataset[variable_key]
    resolution = get_dataset_resolution(model_variable.to_dataset(name="__resolution__"))
    mean_dynamic_topography = _load_mean_dynamic_topography(resolution)
    return model_variable - mean_dynamic_topography - REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT


def _interpolate_vertically(
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
    if len(sorted_depths) > 64:
        raise ValueError("Too many depth levels for Class IV bitmask encoding: maximum supported is 64.")
    sorted_profiles = profiles[sort_order, :]
    valid_masks = ~numpy.isnan(sorted_profiles)
    valid_counts = valid_masks.sum(axis=0)
    enough_points = valid_counts >= MINIMUM_POINTS_FOR_CUBIC_SPLINE
    if not numpy.any(enough_points):
        return result
    eligible_indices = numpy.where(enough_points)[0]
    powers = 2 ** numpy.arange(len(sorted_depths), dtype=numpy.int64)
    mask_ids = valid_masks[:, eligible_indices].astype(numpy.int64).T @ powers
    unique_mask_ids, inverse = numpy.unique(mask_ids, return_inverse=True)
    for group_idx in range(len(unique_mask_ids)):
        group_local = numpy.where(inverse == group_idx)[0]
        indices = eligible_indices[group_local]
        valid_mask = valid_masks[:, indices[0]]
        group_depths = sorted_depths[valid_mask]
        group_targets = target_depths[indices]
        in_range = (group_targets >= group_depths[0]) & (group_targets <= group_depths[-1])
        if not numpy.any(in_range):
            continue
        active_indices = indices[in_range]
        active_targets = group_targets[in_range]
        active_profiles = sorted_profiles[valid_mask][:, active_indices]
        for start in range(0, len(active_indices), VERTICAL_INTERPOLATION_BATCH_SIZE):
            end = min(start + VERTICAL_INTERPOLATION_BATCH_SIZE, len(active_indices))
            batch_idx = active_indices[start:end]
            batch_targets = active_targets[start:end]
            batch_profiles = active_profiles[:, start:end]
            spline = CubicSpline(group_depths, batch_profiles, axis=0, bc_type="natural")
            interpolated = spline(batch_targets)
            if interpolated.ndim == 2:
                result[batch_idx] = interpolated[numpy.arange(len(batch_idx)), numpy.arange(len(batch_idx))]
            else:
                result[batch_idx] = interpolated
    return result


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

    insert_idx = numpy.searchsorted(sorted_depths, target_depths)
    idx_upper = numpy.clip(insert_idx, 0, len(sorted_depths) - 1)
    idx_lower = numpy.clip(insert_idx - 1, 0, len(sorted_depths) - 1)

    exact_mask = sorted_depths[idx_upper] == target_depths
    idx_lower = numpy.where(exact_mask, idx_upper, idx_lower)

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

    invalid = numpy.isnan(lower_values) | numpy.isnan(upper_values)
    result[~invalid] = interpolated[~invalid]
    return result


def _model_data_with_depth_dimension(model_data: xarray.DataArray) -> xarray.DataArray:
    depth_key = Dimension.DEPTH.key()
    if depth_key in model_data.dims:
        return model_data
    return model_data.expand_dims({depth_key: [0.0]})


def _should_use_bracket_vertical_interpolation(variable_key: str) -> bool:
    return variable_key in (
        Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
        Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
        Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
    )


def _horizontally_interpolated_profiles(
    time_slice: xarray.DataArray,
    observation_group: pandas.DataFrame,
) -> numpy.ndarray:
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    observation_latitudes = observation_group[latitude_key].values
    observation_longitudes = observation_group[longitude_key].values
    return time_slice.interp(
        {
            latitude_key: xarray.DataArray(observation_latitudes, dims="observation"),
            longitude_key: xarray.DataArray(observation_longitudes, dims="observation"),
        },
        method="linear",
    ).values


def _interpolated_model_values_for_observation_group(
    time_slice: xarray.DataArray,
    observation_group: pandas.DataFrame,
    model_depths: numpy.ndarray,
    variable_key: str,
) -> numpy.ndarray:
    observation_depths = observation_group[Dimension.DEPTH.key()].values
    horizontally_interpolated = _horizontally_interpolated_profiles(time_slice, observation_group)
    if _should_use_bracket_vertical_interpolation(variable_key):
        return _interpolate_vertically_bracket(
            horizontally_interpolated,
            model_depths,
            observation_depths,
        )
    return _interpolate_vertically(horizontally_interpolated, model_depths, observation_depths)


def _assign_model_values_for_first_day(
    model_values: numpy.ndarray,
    model_data: xarray.DataArray,
    first_day_group: pandas.DataFrame,
    first_day_index: int,
    lead_day_to_index: dict[object, int],
    model_depths: numpy.ndarray,
    variable_key: str,
) -> None:
    for lead_day, observation_group in first_day_group.groupby("lead_day", sort=False):
        time_slice = model_data.isel(
            {
                Dimension.FIRST_DAY_DATETIME.key(): first_day_index,
                Dimension.LEAD_DAY_INDEX.key(): lead_day_to_index[lead_day],
            }
        ).compute()
        model_values[observation_group.index.values] = _interpolated_model_values_for_observation_group(
            time_slice,
            observation_group,
            model_depths,
            variable_key,
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


def _compute_rmsd_table(
    dataframe: pandas.DataFrame,
    variable_key: str,
) -> pandas.DataFrame:
    valid_dataframe = dataframe.dropna(subset=["model_value", "observation_value"])
    grouped = (
        valid_dataframe.assign(
            squared_difference=(valid_dataframe["model_value"] - valid_dataframe["observation_value"]) ** 2
        )
        .groupby(["depth_bin", "lead_day"], as_index=False)
        .agg(
            rmsd=("squared_difference", lambda values: numpy.sqrt(values.mean())), count=("squared_difference", "size")
        )
    )
    grouped["count"] = grouped["count"].astype(int)
    grouped["variable"] = variable_key
    return grouped[["variable", "depth_bin", "lead_day", "rmsd", "count"]]


def _observation_variable_depth_label(standard_name: str, depth_bin: str) -> str:
    display_name, unit = VARIABLE_METADATA[standard_name]
    if standard_name == Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key():
        # TODO: replace the reused SSH standard name with an agreed SLA-specific one for Class IV observations.
        display_name = "sea level anomaly"
    return f"{display_name.capitalize()} ({unit}) [{standard_name}]{{{depth_bin}}}"


def _format_results(results_dataframe: pandas.DataFrame, lead_days_count: int) -> pandas.DataFrame:
    pivot_table = results_dataframe.pivot_table(
        values="rmsd",
        index=["variable", "depth_bin"],
        columns="lead_day",
        aggfunc="first",
    ).reset_index()
    first_available_day = results_dataframe["lead_day"].min()
    observation_counts = results_dataframe[results_dataframe["lead_day"] == first_available_day][
        ["variable", "depth_bin", "count"]
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
    result = pivot_table.set_index("label")[lead_columns].rename(columns=column_rename)
    result.index.name = None
    result.columns.name = None
    return result


def _class4_variable_results(
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


def _formatted_class4_results(
    all_results: list[pandas.DataFrame],
    lead_days_count: int,
) -> pandas.DataFrame:
    if not all_results:
        return pandas.DataFrame()
    return _format_results(pandas.concat(all_results, ignore_index=True), lead_days_count)


def _class4_variable_results(
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


def _formatted_class4_results(
    all_results: list[pandas.DataFrame],
    lead_days_count: int,
) -> pandas.DataFrame:
    if not all_results:
        return pandas.DataFrame()
    return _format_results(pandas.concat(all_results, ignore_index=True), lead_days_count)


def rmsd_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
) -> pandas.DataFrame:
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    lead_days_count = challenger.sizes[Dimension.LEAD_DAY_INDEX.key()]
    observations = reference_dataset
    base_observations_dataframe, selected_observation_indices, observation_dimension_key = (
        _create_observations_base_dataframe(observations, lead_days_count)
    )

    all_results = []
    resolved_variable_keys = [(variable.key(), variable.key(), variable.key()) for variable in variables]

    for standard_variable_key, observation_variable_key, challenger_variable_key in resolved_variable_keys:
        variable_results = _class4_variable_results(
            challenger,
            observations,
            base_observations_dataframe,
            selected_observation_indices,
            observation_dimension_key,
            observation_variable_key,
            challenger_variable_key,
            standard_variable_key,
        )
        if not variable_results.empty:
            all_results.append(variable_results)

    return _formatted_class4_results(all_results, lead_days_count)
