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
    LEAD_DAYS_COUNT,
    VARIABLE_LABELS,
    DEPTH_BINS_DEFAULT,
    DEPTH_BINS_BY_VARIABLE,
    VARIABLE_DISPLAY_ORDER,
    DEPTH_BIN_DISPLAY_ORDER,
)
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.climate_forecast_standard_names import (
    rename_dataset_with_standard_names,
)

REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT = -0.1148
MEAN_SEA_SURFACE_HEIGHT_URL = "https://minio.dive.edito.eu/project-ml-compression/public/glorys12_mssh_2024.zarr"
MINIMUM_POINTS_FOR_CUBIC_SPLINE = 4
VERTICAL_INTERPOLATION_BATCH_SIZE = 1000


def _load_mean_sea_surface_height() -> xarray.DataArray:
    mean_sea_surface_height_dataset = xarray.open_dataset(
        MEAN_SEA_SURFACE_HEIGHT_URL,
        engine="zarr",
        chunks="auto",
    )
    return mean_sea_surface_height_dataset["mssh"]


def _get_depth_bins(variable_key: str) -> dict[str, tuple[float, float]]:
    return DEPTH_BINS_BY_VARIABLE.get(variable_key, DEPTH_BINS_DEFAULT)


def _assign_depth_bins(
    depth_values: numpy.ndarray,
    depth_bins: dict[str, tuple[float, float]],
) -> numpy.ndarray:
    bin_assignments = numpy.full(len(depth_values), "", dtype=object)
    for bin_name, (depth_minimum, depth_maximum) in depth_bins.items():
        mask = (depth_values >= depth_minimum) & (depth_values < depth_maximum)
        bin_assignments[mask] = bin_name
    return bin_assignments


def _create_observations_dataframe(
    observations_dataset: xarray.Dataset,
    observation_variable_key: str,
    standard_variable_key: str,
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
        & (observation_subset["lead_day"] < LEAD_DAYS_COUNT)
    ).compute()
    observation_subset = observation_subset.isel({observation_dimension_key: valid_observation_mask})

    observations_dataframe = observation_subset.compute().to_dataframe().reset_index()
    observations_dataframe = observations_dataframe.drop(columns=[observation_dimension_key], errors="ignore")
    observations_dataframe = observations_dataframe[
        ["observation_value", time_key, latitude_key, longitude_key, "first_day", depth_key, "lead_day"]
    ]

    depth_bins = _get_depth_bins(standard_variable_key)
    observations_dataframe["depth_bin"] = _assign_depth_bins(observations_dataframe[depth_key].values, depth_bins)
    return observations_dataframe.loc[observations_dataframe["depth_bin"] != ""]


def _apply_sea_surface_height_correction(
    dataframe: pandas.DataFrame,
    variable_key: str,
) -> pandas.DataFrame:
    if variable_key != Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key():
        return dataframe
    mean_sea_surface_height = _load_mean_sea_surface_height()
    latitudes = xarray.DataArray(dataframe["latitude"].values, dims="observation")
    longitudes = xarray.DataArray(dataframe["longitude"].values, dims="observation")
    mean_sea_surface_height_at_observations = mean_sea_surface_height.interp(
        latitude=latitudes, longitude=longitudes, method="linear"
    ).values
    corrected_dataframe = dataframe.copy()
    corrected_dataframe["observation_value"] = (
        dataframe["observation_value"]
        + mean_sea_surface_height_at_observations
        + REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT
    )
    return corrected_dataframe


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


def _interpolate_model_to_observations(
    model_data: xarray.DataArray,
    observations_dataframe: pandas.DataFrame,
) -> numpy.ndarray:
    observations_dataframe = observations_dataframe.reset_index(drop=True)
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
        available_lead_days = list(grouped_by_lead_day.groups.keys())
        lead_day_indices = [lead_day_to_index[lead_day] for lead_day in available_lead_days]
        first_day_index = first_day_to_index[first_day]
        first_day_model_subset = model_data.isel(
            {
                Dimension.FIRST_DAY_DATETIME.key(): first_day_index,
                Dimension.LEAD_DAY_INDEX.key(): lead_day_indices,
            }
        ).compute()
        lead_day_to_local_index = {lead_day: local_index for local_index, lead_day in enumerate(available_lead_days)}
        for lead_day, observation_group in grouped_by_lead_day:
            time_slice = first_day_model_subset.isel(
                {Dimension.LEAD_DAY_INDEX.key(): lead_day_to_local_index[lead_day]}
            )
            observation_latitudes = observation_group["latitude"].values
            observation_longitudes = observation_group["longitude"].values
            observation_depths = observation_group["depth"].values
            observation_indices = observation_group.index.values
            horizontally_interpolated = time_slice.interp(
                {
                    Dimension.LATITUDE.key(): xarray.DataArray(observation_latitudes, dims="observation"),
                    Dimension.LONGITUDE.key(): xarray.DataArray(observation_longitudes, dims="observation"),
                },
                method="linear",
            ).values
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


def _format_results(results_dataframe: pandas.DataFrame) -> pandas.DataFrame:
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
    pivot_table["variable_sort"] = pivot_table["variable"].map(VARIABLE_DISPLAY_ORDER)
    pivot_table["depth_sort"] = pivot_table["depth_bin"].map(DEPTH_BIN_DISPLAY_ORDER)
    pivot_table = pivot_table.sort_values(["variable_sort", "depth_sort"]).drop(columns=["variable_sort", "depth_sort"])
    pivot_table["variable"] = pivot_table["variable"].map(VARIABLE_LABELS)
    lead_labels = lead_day_labels(1, LEAD_DAYS_COUNT)
    rename_columns = {
        column: lead_labels[column] for column in pivot_table.columns if isinstance(column, (int, numpy.integer))
    }
    rename_columns["variable"] = "Variable"
    rename_columns["depth_bin"] = "Depth Range"
    rename_columns["count"] = "Number of observations for the first lead day"
    return pivot_table.rename(columns=rename_columns).reset_index(drop=True)


def rmsd_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
) -> pandas.DataFrame:
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    observations = reference_dataset

    all_results = []
    resolved_variables = [(variable.key(), variable.key(), variable.key()) for variable in variables]

    for standard_variable_key, observation_variable_key, challenger_variable_key in resolved_variables:
        observations_dataframe = _create_observations_dataframe(
            observations,
            observation_variable_key,
            standard_variable_key,
        )
        if observations_dataframe.empty:
            continue

        observations_dataframe = _apply_sea_surface_height_correction(observations_dataframe, standard_variable_key)
        observations_dataframe = observations_dataframe.dropna(subset=["observation_value"])

        model_variable = challenger[challenger_variable_key]
        observations_dataframe["model_value"] = _interpolate_model_to_observations(
            model_variable, observations_dataframe
        )

        variable_results = _compute_rmsd_table(observations_dataframe, standard_variable_key)
        if not variable_results.empty:
            all_results.append(variable_results)

    if not all_results:
        result = pandas.DataFrame()
    else:
        final_df = pandas.concat(all_results, ignore_index=True)
        result = _format_results(final_df)

    return result
