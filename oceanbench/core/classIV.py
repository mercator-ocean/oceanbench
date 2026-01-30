# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray
import pandas
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names

LEAD_DAYS_COUNT = 10
SELECTED_LEAD_DAYS_FOR_DISPLAY = [0, 2, 4, 6, 9]

VARIABLE_DISPLAY_NAMES: dict[str, str] = {
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): "Sea surface height",
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): "Temperature",
    Variable.SEA_WATER_SALINITY.key(): "Salinity",
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(): "Zonal current",
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): "Meridional current",
}


def _standardize_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    preserved_coords = {}
    if "first_day_datetime" in dataset.coords:
        preserved_coords["first_day_datetime"] = dataset["first_day_datetime"]

    standardized = rename_dataset_with_standard_names(dataset)

    variable_rename_mapping = {
        "zos": Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
        "thetao": Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
        "so": Variable.SEA_WATER_SALINITY.key(),
        "uo": Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
        "vo": Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
    }

    variables_to_rename = {old: new for old, new in variable_rename_mapping.items() if old in standardized}
    if variables_to_rename:
        standardized = standardized.rename(variables_to_rename)

    for coord_name, coord_values in preserved_coords.items():
        if coord_name not in standardized.coords:
            standardized = standardized.assign_coords({coord_name: coord_values})

    return standardized


def perform_matchup(
    challenger: xarray.Dataset, observations_dataframe: pandas.DataFrame, variable_name: str
) -> pandas.DataFrame:
    matchup_results = []
    run_dates = challenger[Dimension.FIRST_DAY_DATETIME.key()].values

    for run_date in run_dates:
        forecast = challenger.sel({Dimension.FIRST_DAY_DATETIME.key(): run_date})

        for lead_index in range(LEAD_DAYS_COUNT):
            daily_matchup = _match_single_lead_day(
                forecast, observations_dataframe, variable_name, lead_index, run_date
            )
            if daily_matchup is not None:
                matchup_results.append(daily_matchup)

    return pandas.concat(matchup_results, ignore_index=True) if matchup_results else pandas.DataFrame()


def _match_single_lead_day(
    forecast: xarray.Dataset,
    observations_dataframe: pandas.DataFrame,
    variable_name: str,
    lead_index: int,
    run_date,
) -> pandas.DataFrame:
    valid_time = pandas.to_datetime(run_date) + pandas.Timedelta(days=lead_index + 1)
    daily_observations = observations_dataframe[observations_dataframe["time"].dt.date == valid_time.date()].copy()

    if daily_observations.empty:
        return None

    model_slice = forecast.sel({Dimension.LEAD_DAY_INDEX.key(): lead_index}).load()
    daily_observations["model_value"] = _interpolate_model_to_observations(
        model_slice, daily_observations, variable_name
    )
    daily_observations["lead_day"] = lead_index
    daily_observations["run_date"] = run_date

    return daily_observations.dropna(subset=["model_value", variable_name])


def _interpolate_model_to_observations(
    model_slice: xarray.Dataset, observations_dataframe: pandas.DataFrame, variable_name: str
) -> numpy.ndarray:
    horizontal_coordinates = {
        Dimension.LATITUDE.key(): xarray.DataArray(
            observations_dataframe[Dimension.LATITUDE.key()].values, dims="points"
        ),
        Dimension.LONGITUDE.key(): xarray.DataArray(
            observations_dataframe[Dimension.LONGITUDE.key()].values, dims="points"
        ),
    }
    horizontal_interpolation = model_slice[variable_name].interp(horizontal_coordinates, method="linear")

    if _has_vertical_dimension(model_slice[variable_name]) and Dimension.DEPTH.key() in observations_dataframe.columns:
        return _interpolate_vertical(horizontal_interpolation, observations_dataframe[Dimension.DEPTH.key()].values)

    return horizontal_interpolation.values


def _has_vertical_dimension(data_array: xarray.DataArray) -> bool:
    return Dimension.DEPTH.key() in data_array.dims


def _interpolate_vertical(horizontal_interpolation: xarray.DataArray, depths: numpy.ndarray) -> numpy.ndarray:
    model_values = [
        _interpolate_single_depth(horizontal_interpolation, index, depth) for index, depth in enumerate(depths)
    ]
    return pandas.to_numeric(model_values, errors="coerce")


def _interpolate_single_depth(
    horizontal_interpolation: xarray.DataArray, index: int, observation_depth: float
) -> float:
    try:
        result = (
            horizontal_interpolation.isel(points=index)
            .interp({Dimension.DEPTH.key(): observation_depth}, method="cubic")
            .values
        )
        return float(result) if not numpy.isnan(result) else numpy.nan
    except (ValueError, KeyError):
        return numpy.nan


def rmsd_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
    depth_bins_config: dict = None,
) -> pandas.DataFrame:
    challenger_standardized = _standardize_dataset(challenger_dataset)
    reference_standardized = _standardize_dataset(reference_dataset)

    all_results = []

    for variable in variables:
        variable_name = variable.key()

        if variable_name not in reference_standardized:
            continue

        observations_dataframe = _create_observation_dataframe(reference_standardized, variable_name)

        if observations_dataframe.empty:
            continue

        matchup_dataframe = perform_matchup(challenger_standardized, observations_dataframe, variable_name)

        if not matchup_dataframe.empty:
            rmsd_result = rmsd_class4(matchup_dataframe, variable_name)
            all_results.append(rmsd_result)

    if not all_results:
        return pandas.DataFrame()

    return _format_rmsd_table(pandas.concat(all_results, ignore_index=True))


def _create_observation_dataframe(reference_dataset: xarray.Dataset, variable_name: str) -> pandas.DataFrame:
    observations_dataframe = pandas.DataFrame(
        {
            variable_name: reference_dataset[variable_name].values,
            "time": reference_dataset[Dimension.TIME.key()].values,
            Dimension.LATITUDE.key(): reference_dataset[Dimension.LATITUDE.key()].values,
            Dimension.LONGITUDE.key(): reference_dataset[Dimension.LONGITUDE.key()].values,
            Dimension.DEPTH.key(): reference_dataset[Dimension.DEPTH.key()].values,
            "first_day_datetime": reference_dataset["first_day_datetime"].values,
        }
    )
    return observations_dataframe.dropna(subset=[variable_name])


def rmsd_class4(matchup_dataframe: pandas.DataFrame, variable_name: str) -> pandas.DataFrame:
    depth_bins = _get_depth_bins_for_variable(variable_name)
    results = []

    for lead_index in sorted(matchup_dataframe["lead_day"].unique()):
        lead_data = matchup_dataframe[matchup_dataframe["lead_day"] == lead_index]
        results.extend(_compute_rmsd_for_depth_bins(lead_data, variable_name, lead_index, depth_bins))

    return pandas.DataFrame(results)


def _get_depth_bins_for_variable(variable_name: str) -> dict:
    depth_bins_by_variable = {
        Variable.EASTWARD_SEA_WATER_VELOCITY.key(): {"15m": (10, 20)},
        Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): {"15m": (10, 20)},
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): {"surface": (-1, 1)},
    }

    default_depth_bins = {
        "0-5m": (0, 5),
        "5-100m": (5, 100),
        "100-300m": (100, 300),
        "300-600m": (300, 600),
    }

    return depth_bins_by_variable.get(variable_name, default_depth_bins)


def _compute_rmsd_for_depth_bins(
    lead_data: pandas.DataFrame, variable_name: str, lead_index: int, depth_bins: dict
) -> list:
    results = []

    for depth_label, (minimum_depth, maximum_depth) in depth_bins.items():
        depth_data = _filter_by_depth_range(lead_data, minimum_depth, maximum_depth)

        if len(depth_data) > 0:
            rmsd = _calculate_rmsd(depth_data["model_value"], depth_data[variable_name])
            results.append(
                {
                    "variable": variable_name,
                    "depth_bin": depth_label,
                    "lead_day": lead_index,
                    "rmsd": rmsd,
                    "count": len(depth_data),
                }
            )

    return results


def _filter_by_depth_range(data: pandas.DataFrame, minimum_depth: float, maximum_depth: float) -> pandas.DataFrame:
    return data[(data["depth"] >= minimum_depth) & (data["depth"] < maximum_depth)]


def _calculate_rmsd(model_values: numpy.ndarray, observation_values: numpy.ndarray) -> float:
    return numpy.sqrt(numpy.mean((model_values - observation_values) ** 2))


def _create_pivot_table_multi_lead(table: pandas.DataFrame) -> pandas.DataFrame:
    return table.pivot_table(
        values="rmsd", index=["variable", "depth_bin"], columns="lead_day", aggfunc="first"
    ).reset_index()


def _format_rmsd_table(combined: pandas.DataFrame, selected_lead_days: list = None) -> pandas.DataFrame:
    if selected_lead_days is None:
        selected_lead_days = SELECTED_LEAD_DAYS_FOR_DISPLAY

    table = combined[combined["lead_day"].isin(selected_lead_days)]

    if table.empty:
        return pandas.DataFrame()

    pivot = _create_pivot_table_multi_lead(table)
    pivot = _add_observation_counts(pivot, table, selected_lead_days[0])
    pivot = _rename_and_sort_variables(pivot)

    return _create_final_output_table_multi_lead(pivot, selected_lead_days)


def _create_final_output_table_multi_lead(pivot: pandas.DataFrame, selected_lead_days: list) -> pandas.DataFrame:
    output_columns = {
        "Variable": pivot["variable"],
        "Depth Range": pivot["depth_bin"],
    }

    all_lead_labels = lead_day_labels(1, LEAD_DAYS_COUNT)

    for lead_index in selected_lead_days:
        if lead_index in pivot.columns:
            label = all_lead_labels[lead_index]
            output_columns[label] = pivot[lead_index].apply(
                lambda value: f"{value:.3f}" if pandas.notna(value) else "-"
            )

    if "count" in pivot.columns:
        output_columns["Number of observations"] = pivot["count"].apply(
            lambda count: int(count) if pandas.notna(count) else "-"
        )

    return pandas.DataFrame(output_columns)


def _add_observation_counts(pivot: pandas.DataFrame, table: pandas.DataFrame, reference_lead: int) -> pandas.DataFrame:
    counts = (
        table[table["lead_day"] == reference_lead]
        .pivot_table(values="count", index=["variable", "depth_bin"], aggfunc="first")
        .reset_index()
    )

    return pivot.merge(counts, on=["variable", "depth_bin"], how="left")


def _rename_and_sort_variables(pivot: pandas.DataFrame) -> pandas.DataFrame:
    variable_display_order = [
        VARIABLE_DISPLAY_NAMES[Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()],
        VARIABLE_DISPLAY_NAMES[Variable.SEA_WATER_SALINITY.key()],
        VARIABLE_DISPLAY_NAMES[Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()],
        VARIABLE_DISPLAY_NAMES[Variable.EASTWARD_SEA_WATER_VELOCITY.key()],
        VARIABLE_DISPLAY_NAMES[Variable.NORTHWARD_SEA_WATER_VELOCITY.key()],
    ]
    depth_bin_display_order = ["surface", "0-5m", "5-100m", "100-300m", "300-600m", "15m"]

    pivot["variable"] = pivot["variable"].map(VARIABLE_DISPLAY_NAMES)
    pivot["var_sort"] = pivot["variable"].map(
        {variable: index for index, variable in enumerate(variable_display_order)}
    )
    pivot["depth_sort"] = pivot["depth_bin"].map({depth: index for index, depth in enumerate(depth_bin_display_order)})

    return pivot.sort_values(["var_sort", "depth_sort"]).reset_index(drop=True)
