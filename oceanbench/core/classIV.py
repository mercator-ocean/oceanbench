# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray
import pandas
from scipy.interpolate import CubicSpline

from oceanbench.core.dataset_utils import (
    Dimension,
    Variable,
    VARIABLE_LABELS,
    DEPTH_BINS_DEFAULT,
    VARIABLE_DISPLAY_ORDER,
    DEPTH_BIN_DISPLAY_ORDER,
)
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.climate_forecast_standard_names import (
    rename_dataset_with_standard_names,
)
from oceanbench.core.resolution import get_dataset_resolution

REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT = -0.1148
MINIMUM_POINTS_FOR_CUBIC_SPLINE = 4
VERTICAL_INTERPOLATION_BATCH_SIZE = 1000
VELOCITY_TARGET_DEPTH_METERS = 15.0


def _mean_dynamic_topography_zarr_url(resolution: str) -> str:
    if resolution == "twelfth_degree":
        return (
            "https://minio.dive.edito.eu/project-oceanbench/public/glorys12_mean_sea_surface_height_2024/"
            "GLO-MFC_001_030_mdt.zarr"
        )
    if resolution == "quarter_degree":
        return (
            "https://minio.dive.edito.eu/project-oceanbench/public/glorys12_mean_sea_surface_height_2024/"
            "GLO-MFC_001_030_mdt_025deg.zarr"
        )
    if resolution == "one_degree":
        return (
            "https://minio.dive.edito.eu/project-oceanbench/public/glorys12_mean_sea_surface_height_2024/"
            "GLO-MFC_001_030_mdt_1_deg.zarr"
        )
    raise ValueError(f"Unsupported resolution : {resolution}.")


def _load_mean_dynamic_topography(resolution: str) -> xarray.DataArray:
    dataset = xarray.open_dataset(
        _mean_dynamic_topography_zarr_url(resolution),
        engine="zarr",
        chunks="auto",
        consolidated=True,
    )
    dataset = rename_dataset_with_standard_names(dataset)
    dataset = dataset.rename(
        {
            dimension_name: standard_name
            for dimension_name, standard_name in {
                "lat": Dimension.LATITUDE.key(),
                "lon": Dimension.LONGITUDE.key(),
            }.items()
            if dimension_name in dataset.dims
        }
    )
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
    bin_assignments[surface_mask] = "SST"
    return bin_assignments


def _interpolate_observations_to_target_depth(
    observations_dataframe: pandas.DataFrame,
    target_depth: float,
) -> pandas.DataFrame:
    if observations_dataframe.empty:
        return observations_dataframe
    observations_dataframe = observations_dataframe.dropna(subset=["observation_value", Dimension.DEPTH.key()])
    if observations_dataframe.empty:
        return observations_dataframe
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

    records = []
    for group_key, group in observations_dataframe.groupby(group_keys, sort=False):
        depths = group[depth_key].to_numpy()
        values = group["observation_value"].to_numpy()
        below_depths = depths[depths <= target_depth]
        above_depths = depths[depths >= target_depth]
        if below_depths.size == 0 or above_depths.size == 0:
            continue
        below_depth = below_depths.max()
        above_depth = above_depths.min()
        below_value = values[depths == below_depth].mean()
        above_value = values[depths == above_depth].mean()
        if numpy.isclose(below_depth, above_depth):
            interpolated_value = below_value
        else:
            weight = (target_depth - below_depth) / (above_depth - below_depth)
            interpolated_value = below_value + weight * (above_value - below_value)
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        record = dict(zip(group_keys, group_key))
        record.update(
            {
                "observation_value": interpolated_value,
                depth_key: target_depth,
            }
        )
        records.append(record)

    if not records:
        return pandas.DataFrame(columns=target_columns)
    result = pandas.DataFrame.from_records(records)
    return result[target_columns]


def _create_observations_dataframe(
    observations_dataset: xarray.Dataset,
    observation_variable_key: str,
    standard_variable_key: str,
    lead_days_count: int,
) -> pandas.DataFrame:
    time_key = Dimension.TIME.key()
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    depth_key = Dimension.DEPTH.key()
    observation_dimension_key = observations_dataset[observation_variable_key].dims[0]

    selected_variable_keys = [
        observation_variable_key,
        time_key,
        latitude_key,
        longitude_key,
        first_day_key,
        depth_key,
    ]

    observation_subset = observations_dataset[selected_variable_keys].rename(
        {
            observation_variable_key: "observation_value",
            first_day_key: "first_day",
        }
    )

    lead_day = ((observation_subset[time_key] - observation_subset["first_day"]) / numpy.timedelta64(1, "D")).astype(
        "int64"
    ) - 1
    observation_subset = observation_subset.assign(lead_day=lead_day)
    valid_observation_mask = (
        observation_subset["observation_value"].notnull()
        & (observation_subset["lead_day"] >= 0)
        & (observation_subset["lead_day"] < lead_days_count)
    ).compute()
    observation_subset = observation_subset.isel({observation_dimension_key: valid_observation_mask})

    observations_dataframe = observation_subset.compute().to_dataframe().reset_index()
    observations_dataframe = observations_dataframe.drop(columns=[observation_dimension_key], errors="ignore")
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


def _convert_forecast_ssh_to_sla(
    model_variable: xarray.DataArray,
    variable_key: str,
) -> xarray.DataArray:
    if variable_key != Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key():
        return model_variable
    model_dataset = rename_dataset_with_standard_names(model_variable.to_dataset(name=variable_key))
    model_dataset = model_dataset.rename(
        {
            dimension_name: standard_name
            for dimension_name, standard_name in {
                "lat": Dimension.LATITUDE.key(),
                "lon": Dimension.LONGITUDE.key(),
            }.items()
            if dimension_name in model_dataset.dims
        }
    )
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


def _interpolate_model_to_observations(
    model_data: xarray.DataArray,
    observations_dataframe: pandas.DataFrame,
    variable_key: str,
) -> numpy.ndarray:
    observations_dataframe = observations_dataframe.reset_index(drop=True)
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    depth_key = Dimension.DEPTH.key()
    if depth_key not in model_data.dims:
        model_data = model_data.expand_dims({depth_key: [0.0]})
    model_depths = model_data[depth_key].values
    first_days = model_data[Dimension.FIRST_DAY_DATETIME.key()].values
    lead_days = model_data[Dimension.LEAD_DAY_INDEX.key()].values
    first_day_to_index = {first_day: index for index, first_day in enumerate(first_days)}
    lead_day_to_index = {lead_day: index for index, lead_day in enumerate(lead_days)}
    model_values = numpy.full(len(observations_dataframe), numpy.nan)
    grouped_by_first_day = observations_dataframe.groupby("first_day", sort=False)
    for first_day, first_day_group in grouped_by_first_day:
        grouped_by_lead_day = first_day_group.groupby("lead_day", sort=False)
        first_day_index = first_day_to_index[first_day]
        for lead_day, observation_group in grouped_by_lead_day:
            time_slice = model_data.isel(
                {
                    Dimension.FIRST_DAY_DATETIME.key(): first_day_index,
                    Dimension.LEAD_DAY_INDEX.key(): lead_day_to_index[lead_day],
                }
            ).compute()
            observation_latitudes = observation_group[latitude_key].values
            observation_longitudes = observation_group[longitude_key].values
            observation_depths = observation_group[depth_key].values
            observation_indices = observation_group.index.values
            horizontally_interpolated = time_slice.interp(
                {
                    latitude_key: xarray.DataArray(observation_latitudes, dims="observation"),
                    longitude_key: xarray.DataArray(observation_longitudes, dims="observation"),
                },
                method="linear",
            ).values
            if variable_key in (
                Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
                Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
                Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
                Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
            ):
                interpolated = _interpolate_vertically_bracket(
                    horizontally_interpolated,
                    model_depths,
                    observation_depths,
                )
            else:
                interpolated = _interpolate_vertically(horizontally_interpolated, model_depths, observation_depths)
            model_values[observation_indices] = interpolated
    return model_values


def _compute_rmsd_table(
    dataframe: pandas.DataFrame,
    variable_key: str,
) -> pandas.DataFrame:
    valid_dataframe = dataframe.dropna(subset=["model_value", "observation_value"])

    def compute_group_rmsd(group):
        differences = group["model_value"].values - group["observation_value"].values
        return pandas.Series(
            {
                "rmsd": numpy.sqrt(numpy.mean(differences**2)),
                "count": len(group),
            }
        )

    grouped = (
        valid_dataframe.groupby(["depth_bin", "lead_day"]).apply(compute_group_rmsd, include_groups=False).reset_index()
    )
    grouped["count"] = grouped["count"].astype(int)
    grouped["variable"] = variable_key
    return grouped[["variable", "depth_bin", "lead_day", "rmsd", "count"]]


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
    sst_sort_mask = (pivot_table["variable"] == Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()) & (
        pivot_table["depth_bin"] == "SST"
    )
    pivot_table.loc[sst_sort_mask, "variable_sort"] = VARIABLE_DISPLAY_ORDER[Variable.SEA_WATER_SALINITY.key()] + 0.5
    pivot_table["depth_sort"] = pivot_table["depth_bin"].map(DEPTH_BIN_DISPLAY_ORDER)
    pivot_table = pivot_table.sort_values(["variable_sort", "depth_sort"]).drop(columns=["variable_sort", "depth_sort"])
    pivot_table["variable"] = pivot_table["variable"].map(VARIABLE_LABELS)

    sst_display_mask = (pivot_table["variable"] == "temperature") & (pivot_table["depth_bin"] == "SST")
    pivot_table.loc[sst_display_mask, "variable"] = "surface temperature"
    pivot_table.loc[sst_display_mask, "depth_bin"] = "surface"
    sla_display_mask = pivot_table["variable"] == "surface height"
    pivot_table.loc[sla_display_mask, "variable"] = "sea level anomaly"

    lead_labels = lead_day_labels(1, lead_days_count)
    rename_columns = {
        column: lead_labels[column] for column in pivot_table.columns if isinstance(column, (int, numpy.integer))
    }
    rename_columns["variable"] = "Variable"
    rename_columns["depth_bin"] = "Depth Range"
    rename_columns["count"] = "Number of observations at lead day 1"
    pivot_table = pivot_table.rename(columns=rename_columns).reset_index(drop=True)
    lead_day_columns = lead_labels
    ordered_columns = [
        "Variable",
        "Depth Range",
        "Number of observations at lead day 1",
        *lead_day_columns,
    ]
    ordered_columns = [column for column in ordered_columns if column in pivot_table.columns]
    return pivot_table[ordered_columns]


def rmsd_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
) -> pandas.DataFrame:
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    lead_days_count = challenger.sizes[Dimension.LEAD_DAY_INDEX.key()]
    observations = reference_dataset

    all_results = []
    resolved_variables = [(variable.key(), variable.key(), variable.key()) for variable in variables]

    for standard_variable_key, observation_variable_key, challenger_variable_key in resolved_variables:
        observations_dataframe = _create_observations_dataframe(
            observations,
            observation_variable_key,
            standard_variable_key,
            lead_days_count,
        )
        if observations_dataframe.empty:
            continue

        observations_dataframe = observations_dataframe.dropna(subset=["observation_value"])

        model_variable = _convert_forecast_ssh_to_sla(
            challenger[challenger_variable_key],
            standard_variable_key,
        )
        observations_dataframe["model_value"] = _interpolate_model_to_observations(
            model_variable, observations_dataframe, standard_variable_key
        )

        variable_results = _compute_rmsd_table(observations_dataframe, standard_variable_key)
        if not variable_results.empty:
            all_results.append(variable_results)

    if not all_results:
        result = pandas.DataFrame()
    else:
        final_df = pandas.concat(all_results, ignore_index=True)
        result = _format_results(final_df, lead_days_count)

    return result
