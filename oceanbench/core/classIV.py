# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Class IV validation module.

Computes RMSD between model forecasts and in-situ observations using
scipy interpolation for efficient batch processing.
"""

import time
import numpy
import xarray
import pandas
from scipy.interpolate import RegularGridInterpolator, CubicSpline

from oceanbench.core.dataset_utils import (
    Dimension,
    Variable,
    VARIABLE_LABELS,
    DEPTH_BINS_DEFAULT,
    DEPTH_BINS_BY_VARIABLE,
)
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.climate_forecast_standard_names import (
    rename_dataset_with_standard_names,
)

LEAD_DAYS_COUNT = 10
SELECTED_LEAD_DAYS_FOR_DISPLAY = [0, 2, 4, 6, 9]
REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT = -0.1148
MEAN_SEA_SURFACE_HEIGHT_URL = "https://minio.dive.edito.eu/project-ml-compression/public/glorys12_mssh_2024.zarr"

_MEAN_SEA_SURFACE_HEIGHT_CACHE = None


def _load_mean_sea_surface_height() -> xarray.DataArray:
    """Load mean sea surface height with caching."""
    global _MEAN_SEA_SURFACE_HEIGHT_CACHE
    if _MEAN_SEA_SURFACE_HEIGHT_CACHE is None:
        dataset = xarray.open_dataset(MEAN_SEA_SURFACE_HEIGHT_URL, engine="zarr", chunks=None)
        _MEAN_SEA_SURFACE_HEIGHT_CACHE = dataset["mssh"].load()
    return _MEAN_SEA_SURFACE_HEIGHT_CACHE


COORDINATE_NAME_MAPPING = {
    "lat": Dimension.LATITUDE.key(),
    "lon": Dimension.LONGITUDE.key(),
}

# Mapping from short variable names to CF standard names
VARIABLE_NAME_MAPPING = {
    "zos": Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
    "thetao": Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
    "so": Variable.SEA_WATER_SALINITY.key(),
    "uo": Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
    "vo": Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
}


def _standardize_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    """Standardize variable and coordinate names."""
    dataset = rename_dataset_with_standard_names(dataset)

    # Rename coordinates (lat -> latitude, lon -> longitude)
    coord_rename = {k: v for k, v in COORDINATE_NAME_MAPPING.items() if k in dataset.dims}
    if coord_rename:
        dataset = dataset.rename(coord_rename)

    # Rename variables that don't have standard_name attribute (e.g., observations)
    var_rename = {k: v for k, v in VARIABLE_NAME_MAPPING.items() if k in dataset}
    if var_rename:
        dataset = dataset.rename(var_rename)

    return dataset


def _get_depth_bins(variable_key: str) -> dict[str, tuple[float, float]]:
    """Get depth bins for a variable."""
    return DEPTH_BINS_BY_VARIABLE.get(variable_key, DEPTH_BINS_DEFAULT)


def _create_observations_dataframe(
    observations_dataset: xarray.Dataset,
    variable_key: str,
) -> pandas.DataFrame:
    """Extract observations as a DataFrame with all required columns."""
    dataframe = pandas.DataFrame(
        {
            "observation_value": observations_dataset[variable_key].values,
            "time": observations_dataset.time.values,
            "latitude": observations_dataset[Dimension.LATITUDE.key()].values,
            "longitude": observations_dataset[Dimension.LONGITUDE.key()].values,
            "first_day": observations_dataset[Dimension.FIRST_DAY_DATETIME.key()].values,
        }
    )

    if Dimension.DEPTH.key() in observations_dataset:
        dataframe["depth"] = observations_dataset[Dimension.DEPTH.key()].values
    else:
        dataframe["depth"] = 0.0

    dataframe = dataframe.dropna(subset=["observation_value"])

    dataframe["lead_day"] = (
        pandas.to_datetime(dataframe["time"]) - pandas.to_datetime(dataframe["first_day"])
    ).dt.days - 1

    depth_bins = _get_depth_bins(variable_key)
    dataframe["depth_bin"] = ""
    for bin_name, (depth_min, depth_max) in depth_bins.items():
        mask = (dataframe["depth"] >= depth_min) & (dataframe["depth"] < depth_max)
        dataframe.loc[mask, "depth_bin"] = bin_name

    return dataframe[dataframe["depth_bin"] != ""]


def _apply_sea_surface_height_correction(
    dataframe: pandas.DataFrame,
    variable_key: str,
) -> pandas.DataFrame:
    """Apply SSH correction for sea surface height observations."""
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


def _interpolate_model_to_observations(
    model_data: xarray.DataArray,
    observations_dataframe: pandas.DataFrame,
) -> numpy.ndarray:
    """
    Interpolate model data to observation locations.

    Uses linear interpolation for latitude/longitude and cubic spline for depth.
    """
    has_depth = Dimension.DEPTH.key() in model_data.dims

    latitudes = model_data[Dimension.LATITUDE.key()].values
    longitudes = model_data[Dimension.LONGITUDE.key()].values
    depths = model_data[Dimension.DEPTH.key()].values if has_depth else None
    first_days = model_data[Dimension.FIRST_DAY_DATETIME.key()].values
    lead_days = model_data[Dimension.LEAD_DAY_INDEX.key()].values

    first_day_to_index = {first_day: index for index, first_day in enumerate(first_days)}
    lead_day_to_index = {lead_day: index for index, lead_day in enumerate(lead_days)}

    model_values = numpy.full(len(observations_dataframe), numpy.nan)

    grouped_observations = observations_dataframe.groupby(["first_day", "lead_day"])

    for (first_day, lead_day), observation_group in grouped_observations:
        if first_day not in first_day_to_index or lead_day not in lead_day_to_index:
            continue

        first_day_index = first_day_to_index[first_day]
        lead_day_index = lead_day_to_index[lead_day]

        model_slice = (
            model_data.isel(
                {
                    Dimension.FIRST_DAY_DATETIME.key(): first_day_index,
                    Dimension.LEAD_DAY_INDEX.key(): lead_day_index,
                }
            )
            .compute()
            .values
        )

        observation_latitudes = observation_group["latitude"].values
        observation_longitudes = observation_group["longitude"].values
        observation_indices = observation_group.index.values

        if has_depth:
            observation_depths = observation_group["depth"].values
            observation_count = len(observation_depths)
            depth_count = len(depths)

            profiles_at_observations = numpy.empty((depth_count, observation_count))

            for depth_index in range(depth_count):
                field_2d = model_slice[depth_index, :, :]
                interpolator = RegularGridInterpolator(
                    (latitudes, longitudes),
                    field_2d,
                    method="linear",
                    bounds_error=False,
                    fill_value=numpy.nan,
                )
                points = numpy.column_stack([observation_latitudes, observation_longitudes])
                profiles_at_observations[depth_index, :] = interpolator(points)

            interpolated_values = numpy.empty(observation_count)
            for observation_index in range(observation_count):
                profile = profiles_at_observations[:, observation_index]
                target_depth = observation_depths[observation_index]

                valid_mask = ~numpy.isnan(profile)
                if numpy.sum(valid_mask) >= 4:
                    valid_depths = depths[valid_mask]
                    valid_profile = profile[valid_mask]

                    sort_indices = numpy.argsort(valid_depths)
                    valid_depths = valid_depths[sort_indices]
                    valid_profile = valid_profile[sort_indices]

                    if target_depth < valid_depths[0] or target_depth > valid_depths[-1]:
                        interpolated_values[observation_index] = numpy.nan
                    else:
                        try:
                            spline = CubicSpline(valid_depths, valid_profile, bc_type="natural")
                            interpolated_values[observation_index] = spline(target_depth)
                        except ValueError:
                            interpolated_values[observation_index] = numpy.nan
                else:
                    interpolated_values[observation_index] = numpy.nan

            model_values[observations_dataframe.index.get_indexer(observation_indices)] = interpolated_values
        else:
            interpolator = RegularGridInterpolator(
                (latitudes, longitudes),
                model_slice,
                method="linear",
                bounds_error=False,
                fill_value=numpy.nan,
            )
            points = numpy.column_stack([observation_latitudes, observation_longitudes])
            model_values[observations_dataframe.index.get_indexer(observation_indices)] = interpolator(points)

    return model_values


def _compute_rmsd_table(
    dataframe: pandas.DataFrame,
    variable_key: str,
) -> pandas.DataFrame:
    """Compute RMSD grouped by depth bin and lead day."""
    dataframe = dataframe.copy()
    dataframe["squared_error"] = (dataframe["model_value"] - dataframe["observation_value"]) ** 2

    dataframe = dataframe.dropna(subset=["model_value"])

    grouped = (
        dataframe.groupby(["depth_bin", "lead_day"])
        .agg(
            squared_error_sum=("squared_error", "sum"),
            count=("squared_error", "count"),
        )
        .reset_index()
    )

    grouped["rmsd"] = numpy.sqrt(grouped["squared_error_sum"] / grouped["count"])
    grouped["variable"] = variable_key

    return grouped[["variable", "depth_bin", "lead_day", "rmsd", "count"]]


def _format_results(results_dataframe: pandas.DataFrame) -> pandas.DataFrame:
    """Format results into final pivot table with display names."""
    if results_dataframe.empty:
        return pandas.DataFrame()

    filtered_dataframe = results_dataframe[results_dataframe["lead_day"].isin(SELECTED_LEAD_DAYS_FOR_DISPLAY)]

    if filtered_dataframe.empty:
        return pandas.DataFrame()

    pivot_table = filtered_dataframe.pivot_table(
        values="rmsd",
        index=["variable", "depth_bin"],
        columns="lead_day",
        aggfunc="first",
    ).reset_index()

    first_lead_day = SELECTED_LEAD_DAYS_FOR_DISPLAY[0]
    observation_counts = filtered_dataframe[filtered_dataframe["lead_day"] == first_lead_day][
        ["variable", "depth_bin", "count"]
    ]
    pivot_table = pivot_table.merge(observation_counts, on=["variable", "depth_bin"], how="left")

    pivot_table["variable"] = pivot_table["variable"].map(VARIABLE_LABELS)

    variable_order = ["temperature", "salinity", "surface height", "eastward velocity", "northward velocity"]
    depth_order = ["surface", "0-5m", "5-100m", "100-300m", "300-600m", "15m"]

    pivot_table["variable_sort"] = pivot_table["variable"].map(
        {variable: index for index, variable in enumerate(variable_order)}
    )
    pivot_table["depth_sort"] = pivot_table["depth_bin"].map({depth: index for index, depth in enumerate(depth_order)})
    pivot_table = pivot_table.sort_values(["variable_sort", "depth_sort"]).drop(columns=["variable_sort", "depth_sort"])

    lead_labels = lead_day_labels(1, LEAD_DAYS_COUNT)
    rename_columns = {
        lead_day: lead_labels[lead_day]
        for lead_day in SELECTED_LEAD_DAYS_FOR_DISPLAY
        if lead_day in pivot_table.columns
    }
    rename_columns["variable"] = "Variable"
    rename_columns["depth_bin"] = "Depth Range"
    rename_columns["count"] = "Number of observations"

    return pivot_table.rename(columns=rename_columns).reset_index(drop=True)


def rmsd_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
) -> pandas.DataFrame:
    """
    Compute RMSD between model forecasts and in-situ observations.

    Uses linear interpolation for latitude/longitude and cubic spline for depth.
    """
    start_time = time.time()

    print("Standardizing datasets...", flush=True)
    challenger = _standardize_dataset(challenger_dataset)
    observations = _standardize_dataset(reference_dataset)

    if challenger.chunks:
        challenger = challenger.compute()

    all_results = []
    variable_keys = [variable.key() for variable in variables if variable.key() in observations]

    print(f"Processing {len(variable_keys)} variables...", flush=True)

    for variable_key in variable_keys:
        print(f"  Processing {variable_key}...", flush=True)

        observations_dataframe = _create_observations_dataframe(observations, variable_key)

        if observations_dataframe.empty:
            continue

        print(f"    > {len(observations_dataframe)} observations", flush=True)

        observations_dataframe = _apply_sea_surface_height_correction(observations_dataframe, variable_key)

        model_variable = challenger[variable_key]
        observations_dataframe["model_value"] = _interpolate_model_to_observations(
            model_variable, observations_dataframe
        )

        variable_results = _compute_rmsd_table(observations_dataframe, variable_key)

        if not variable_results.empty:
            all_results.append(variable_results)

    if not all_results:
        return pandas.DataFrame()

    result = _format_results(pandas.concat(all_results, ignore_index=True))

    elapsed_time = time.time() - start_time
    print(f"✓ Validation complete! (took {elapsed_time:.2f}s)", flush=True)

    return result
