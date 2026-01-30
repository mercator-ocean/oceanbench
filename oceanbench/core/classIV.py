# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray
import pandas
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.lead_day_utils import lead_day_labels

LEAD_DAYS_COUNT = 10
SELECTED_LEAD_DAYS_FOR_DISPLAY = [0, 2, 4, 6, 9]


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
        "lat": xarray.DataArray(observations_dataframe["latitude"].values, dims="points"),
        "lon": xarray.DataArray(observations_dataframe["longitude"].values, dims="points"),
    }
    horizontal_interpolation = model_slice[variable_name].interp(horizontal_coordinates, method="linear")

    if _has_vertical_dimension(model_slice[variable_name]) and "depth" in observations_dataframe.columns:
        return _interpolate_vertical(horizontal_interpolation, observations_dataframe["depth"].values)

    return horizontal_interpolation.values


def _has_vertical_dimension(data_array: xarray.DataArray) -> bool:
    return "depth" in data_array.dims


def _interpolate_vertical(horizontal_interpolation: xarray.DataArray, depths: numpy.ndarray) -> numpy.ndarray:
    model_values = [
        _interpolate_single_depth(horizontal_interpolation, index, depth) for index, depth in enumerate(depths)
    ]
    return pandas.to_numeric(model_values, errors="coerce")


def _interpolate_single_depth(
    horizontal_interpolation: xarray.DataArray, index: int, observation_depth: float
) -> float:
    try:
        result = horizontal_interpolation.isel(points=index).interp(depth=observation_depth, method="linear").values
        return float(result) if not numpy.isnan(result) else numpy.nan
    except (ValueError, KeyError):
        return numpy.nan


def rmsd_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list,
    depth_bins_config: dict = None,
) -> pandas.DataFrame:
    all_results = []

    for variable in variables:
        variable_name = _get_standard_variable_name(variable)

        if variable_name not in reference_dataset:
            continue

        observations_dataframe = _create_observation_dataframe(reference_dataset, variable_name)

        if observations_dataframe.empty:
            continue

        matchup_dataframe = perform_matchup(challenger_dataset, observations_dataframe, variable_name)

        if not matchup_dataframe.empty:
            rmsd_result = rmsd_class4(matchup_dataframe, variable_name)
            all_results.append(rmsd_result)

    if not all_results:
        return pandas.DataFrame()

    return _format_rmsd_table(pandas.concat(all_results, ignore_index=True))


def _get_standard_variable_name(variable: Variable) -> str:
    standard_names = {
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID: "zos",
        Variable.SEA_WATER_POTENTIAL_TEMPERATURE: "thetao",
        Variable.SEA_WATER_SALINITY: "so",
        Variable.EASTWARD_SEA_WATER_VELOCITY: "uo",
        Variable.NORTHWARD_SEA_WATER_VELOCITY: "vo",
    }
    return standard_names[variable]


def _create_observation_dataframe(reference_dataset: xarray.Dataset, variable_name: str) -> pandas.DataFrame:
    observations_dataframe = pandas.DataFrame(
        {
            variable_name: reference_dataset[variable_name].values,
            "time": reference_dataset["time"].values,
            "latitude": reference_dataset["latitude"].values,
            "longitude": reference_dataset["longitude"].values,
            "depth": reference_dataset["depth"].values,
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
        "uo": {"15m": (10, 20)},
        "vo": {"15m": (10, 20)},
        "zos": {"surface": (-1, 1)},
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
    variable_display_names = {
        "zos": "SSH",
        "thetao": "Temperature",
        "so": "Salinity",
        "uo": "Zonal current",
        "vo": "Meridional current",
    }
    variable_display_order = ["Temperature", "Salinity", "SSH", "Zonal current", "Meridional current"]
    depth_bin_display_order = ["surface", "0-5m", "5-100m", "100-300m", "300-600m", "15m"]

    pivot["variable"] = pivot["variable"].map(variable_display_names)
    pivot["var_sort"] = pivot["variable"].map(
        {variable: index for index, variable in enumerate(variable_display_order)}
    )
    pivot["depth_sort"] = pivot["depth_bin"].map({depth: index for index, depth in enumerate(depth_bin_display_order)})

    return pivot.sort_values(["var_sort", "depth_sort"]).reset_index(drop=True)
