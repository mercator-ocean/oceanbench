# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import time
import numpy
import xarray
import pandas
from scipy.interpolate import CubicSpline

from oceanbench.core.dataset_utils import (
    Dimension,
    Variable,
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

LEAD_DAYS_COUNT = 10
REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT = -0.1148
MEAN_SEA_SURFACE_HEIGHT_URL = "https://minio.dive.edito.eu/project-ml-compression/public/glorys12_mssh_2024.zarr"
MINIMUM_POINTS_FOR_CUBIC_SPLINE = 4

_MEAN_SEA_SURFACE_HEIGHT_CACHE = None


def _load_mean_sea_surface_height() -> xarray.DataArray:
    global _MEAN_SEA_SURFACE_HEIGHT_CACHE
    if _MEAN_SEA_SURFACE_HEIGHT_CACHE is None:
        ds = xarray.open_dataset(
            MEAN_SEA_SURFACE_HEIGHT_URL,
            engine="zarr",
            chunks="auto",
        )
        _MEAN_SEA_SURFACE_HEIGHT_CACHE = ds["mssh"]
    return _MEAN_SEA_SURFACE_HEIGHT_CACHE


COORDINATE_NAME_MAPPING = {
    "lat": Dimension.LATITUDE.key(),
    "lon": Dimension.LONGITUDE.key(),
}

VARIABLE_NAME_MAPPING = {
    "zos": Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
    "thetao": Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
    "so": Variable.SEA_WATER_SALINITY.key(),
    "uo": Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
    "vo": Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
}


def _standardize_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    dataset = rename_dataset_with_standard_names(dataset)
    coord_rename = {k: v for k, v in COORDINATE_NAME_MAPPING.items() if k in dataset.dims}
    if coord_rename:
        dataset = dataset.rename(coord_rename)
    var_rename = {k: v for k, v in VARIABLE_NAME_MAPPING.items() if k in dataset}
    if var_rename:
        dataset = dataset.rename(var_rename)
    return dataset


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
    variable_key: str,
) -> pandas.DataFrame:
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
    dataframe["depth_bin"] = _assign_depth_bins(dataframe["depth"].values, depth_bins)
    return dataframe[dataframe["depth_bin"] != ""]


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
    for observation_index in range(observation_count):
        profile = profiles[:, observation_index]
        valid_mask = ~numpy.isnan(profile)
        if numpy.sum(valid_mask) < MINIMUM_POINTS_FOR_CUBIC_SPLINE:
            continue
        valid_depths = model_depths[valid_mask]
        valid_values = profile[valid_mask]
        sort_indices = numpy.argsort(valid_depths)
        sorted_depths = valid_depths[sort_indices]
        sorted_values = valid_values[sort_indices]
        target = target_depths[observation_index]
        if sorted_depths[0] <= target <= sorted_depths[-1]:
            spline = CubicSpline(sorted_depths, sorted_values, bc_type="natural")
            result[observation_index] = spline(target)
    return result


def _interpolate_model_to_observations(
    model_data: xarray.DataArray,
    observations_dataframe: pandas.DataFrame,
) -> numpy.ndarray:
    depth_key = Dimension.DEPTH.key()
    if depth_key not in model_data.dims:
        model_data = model_data.expand_dims({depth_key: [0.0]})
    model_depths = model_data[depth_key].values
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
        time_slice = model_data.isel(
            {
                Dimension.FIRST_DAY_DATETIME.key(): first_day_index,
                Dimension.LEAD_DAY_INDEX.key(): lead_day_index,
            }
        ).compute()
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
        model_values[observations_dataframe.index.get_indexer(observation_indices)] = interpolated
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
    if results_dataframe.empty:
        return pandas.DataFrame()
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
    rename_columns = {col: lead_labels[col] for col in pivot_table.columns if isinstance(col, (int, numpy.integer))}
    rename_columns["variable"] = "Variable"
    rename_columns["depth_bin"] = "Depth Range"
    rename_columns["count"] = "Number of observations"
    return pivot_table.rename(columns=rename_columns).reset_index(drop=True)


def rmsd_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
) -> pandas.DataFrame:
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
        observations_dataframe = observations_dataframe.dropna(subset=["observation_value"])
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
    print(f"Validation complete! (took {elapsed_time:.2f}s)", flush=True)
    return result
