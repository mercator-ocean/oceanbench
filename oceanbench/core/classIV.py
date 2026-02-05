# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import time
import numpy
import xarray
import pandas
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names

LEAD_DAYS_COUNT = 10
SELECTED_LEAD_DAYS_FOR_DISPLAY = [0, 2, 4, 6, 9]
REANALYSIS_MSSH_SHIFT = -0.1148
MSSH_URL = "https://minio.dive.edito.eu/project-ml-compression/public/glorys12_mssh_2024.zarr"

VARIABLE_DISPLAY_NAMES: dict[str, str] = {
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): "SSH",
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): "Temperature",
    Variable.SEA_WATER_SALINITY.key(): "Salinity",
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(): "Zonal current",
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): "Meridional current",
}

DEPTH_BINS_DEFAULT = {
    "0-5m": (0, 5),
    "5-100m": (5, 100),
    "100-300m": (100, 300),
    "300-600m": (300, 600),
}

DEPTH_BINS_BY_VARIABLE = {
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(): {"15m": (10, 20)},
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): {"15m": (10, 20)},
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): {"surface": (-1, 1)},
}

DEPTH_BIN_DISPLAY_ORDER = ["surface", "0-5m", "5-100m", "100-300m", "300-600m", "15m"]

VARIABLE_DISPLAY_ORDER = [
    VARIABLE_DISPLAY_NAMES[Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()],
    VARIABLE_DISPLAY_NAMES[Variable.SEA_WATER_SALINITY.key()],
    VARIABLE_DISPLAY_NAMES[Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()],
    VARIABLE_DISPLAY_NAMES[Variable.EASTWARD_SEA_WATER_VELOCITY.key()],
    VARIABLE_DISPLAY_NAMES[Variable.NORTHWARD_SEA_WATER_VELOCITY.key()],
]

_MSSH_CACHE = None


def rmsd_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
    use_ssh_correction: bool = True,
) -> pandas.DataFrame:
    start_time = time.time()

    print("Standardizing datasets...", flush=True)
    challenger_standardized = _standardize_dataset(challenger_dataset)
    reference_standardized = _standardize_dataset(reference_dataset)

    ssh_variable = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
    if use_ssh_correction and ssh_variable in reference_standardized:
        print("Applying SLA to SSH correction...", flush=True)
        mean_ssh = _load_mean_ssh()
        reference_standardized = _convert_sla_to_ssh(reference_standardized, mean_ssh)

    print("Processing variables...", flush=True)
    all_results = _process_all_variables(challenger_standardized, reference_standardized, variables)

    if not all_results:
        return pandas.DataFrame()

    print("Formatting results...", flush=True)
    result = _format_rmsd_table(pandas.concat(all_results, ignore_index=True))

    elapsed_time = time.time() - start_time
    print(f"Class-4 validation completed in {elapsed_time:.1f} seconds", flush=True)

    return result


def _load_mean_ssh() -> xarray.DataArray:
    global _MSSH_CACHE

    if _MSSH_CACHE is not None:
        return _MSSH_CACHE

    mssh_dataset = xarray.open_dataset(MSSH_URL, engine="zarr", chunks=None)
    _MSSH_CACHE = mssh_dataset["mssh"].load()

    return _MSSH_CACHE


def _convert_sla_to_ssh(
    reference_dataset: xarray.Dataset,
    mean_ssh: xarray.DataArray,
) -> xarray.Dataset:
    ssh_variable = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()

    mssh_at_observations = mean_ssh.interp(
        {
            Dimension.LATITUDE.key(): reference_dataset[Dimension.LATITUDE.key()],
            Dimension.LONGITUDE.key(): reference_dataset[Dimension.LONGITUDE.key()],
        },
        method="linear",
    ).compute()

    reference_dataset[ssh_variable] = reference_dataset[ssh_variable] + mssh_at_observations + REANALYSIS_MSSH_SHIFT

    return reference_dataset


def _standardize_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    preserved_coords = {}
    if "first_day_datetime" in dataset.coords:
        preserved_coords["first_day_datetime"] = dataset["first_day_datetime"]

    standardized = rename_dataset_with_standard_names(dataset)
    standardized = _rename_variables_to_standard_names(standardized)

    for coord_name, coord_values in preserved_coords.items():
        if coord_name not in standardized.coords:
            standardized = standardized.assign_coords({coord_name: coord_values})

    return standardized


def _rename_variables_to_standard_names(dataset: xarray.Dataset) -> xarray.Dataset:
    variable_rename_mapping = {
        "zos": Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
        "thetao": Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
        "so": Variable.SEA_WATER_SALINITY.key(),
        "uo": Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
        "vo": Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
    }

    variables_to_rename = {old: new for old, new in variable_rename_mapping.items() if old in dataset}

    if variables_to_rename:
        return dataset.rename(variables_to_rename)

    return dataset


def _process_all_variables(
    challenger_standardized: xarray.Dataset,
    reference_standardized: xarray.Dataset,
    variables: list[Variable],
) -> list[pandas.DataFrame]:
    all_results = []

    for variable in variables:
        variable_result = _process_single_variable(challenger_standardized, reference_standardized, variable)
        if variable_result is not None:
            all_results.append(variable_result)

    return all_results


def _process_single_variable(
    challenger_standardized: xarray.Dataset,
    reference_standardized: xarray.Dataset,
    variable: Variable,
) -> pandas.DataFrame | None:
    variable_name = variable.key()

    if variable_name not in reference_standardized:
        return None

    observations_dataframe = _create_observation_dataframe(reference_standardized, variable_name)

    if observations_dataframe.empty:
        return None

    matchup_dataframe = _perform_matchup(challenger_standardized, observations_dataframe, variable_name)

    if matchup_dataframe.empty:
        return None

    return _compute_rmsd_class4(matchup_dataframe, variable_name)


def _create_observation_dataframe(
    reference_dataset: xarray.Dataset,
    variable_name: str,
) -> pandas.DataFrame:
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


def _perform_matchup(
    challenger: xarray.Dataset,
    observations_dataframe: pandas.DataFrame,
    variable_name: str,
) -> pandas.DataFrame:
    matchup_results = []
    run_dates = challenger[Dimension.FIRST_DAY_DATETIME.key()].values

    for run_date in run_dates:
        forecast = challenger.sel({Dimension.FIRST_DAY_DATETIME.key(): run_date})
        run_matchups = _perform_matchup_for_single_run(forecast, observations_dataframe, variable_name, run_date)
        matchup_results.extend(run_matchups)

    if not matchup_results:
        return pandas.DataFrame()

    return pandas.concat(matchup_results, ignore_index=True)


def _perform_matchup_for_single_run(
    forecast: xarray.Dataset,
    observations_dataframe: pandas.DataFrame,
    variable_name: str,
    run_date: numpy.datetime64,
) -> list[pandas.DataFrame]:
    run_matchups = []

    for lead_index in range(LEAD_DAYS_COUNT):
        daily_matchup = _match_single_lead_day(forecast, observations_dataframe, variable_name, lead_index, run_date)
        if daily_matchup is not None:
            run_matchups.append(daily_matchup)

    return run_matchups


def _match_single_lead_day(
    forecast: xarray.Dataset,
    observations_dataframe: pandas.DataFrame,
    variable_name: str,
    lead_index: int,
    run_date: numpy.datetime64,
) -> pandas.DataFrame | None:
    valid_time = pandas.to_datetime(run_date) + pandas.Timedelta(days=lead_index + 1)
    daily_observations = observations_dataframe[observations_dataframe["time"].dt.date == valid_time.date()].copy()

    if daily_observations.empty:
        return None

    model_slice = forecast.sel({Dimension.LEAD_DAY_INDEX.key(): lead_index})
    daily_observations["model_value"] = _interpolate_model_to_observations(
        model_slice, daily_observations, variable_name
    )
    daily_observations["lead_day"] = lead_index
    daily_observations["run_date"] = run_date

    return daily_observations.dropna(subset=["model_value", variable_name])


def _interpolate_model_to_observations(
    model_slice: xarray.Dataset,
    observations_dataframe: pandas.DataFrame,
    variable_name: str,
) -> numpy.ndarray:
    data_array = model_slice[variable_name]

    lats = xarray.DataArray(observations_dataframe[Dimension.LATITUDE.key()].values, dims="points")
    lons = xarray.DataArray(observations_dataframe[Dimension.LONGITUDE.key()].values, dims="points")

    horizontal_interpolation = data_array.interp(
        {Dimension.LATITUDE.key(): lats, Dimension.LONGITUDE.key(): lons},
        method="linear",
    )

    if Dimension.DEPTH.key() not in data_array.dims:
        return horizontal_interpolation.values

    if Dimension.DEPTH.key() in observations_dataframe.columns:
        depths = xarray.DataArray(observations_dataframe[Dimension.DEPTH.key()].values, dims="points")
        result = horizontal_interpolation.interp({Dimension.DEPTH.key(): depths}, method="cubic")
        return result.values

    return horizontal_interpolation.values


def _compute_rmsd_class4(
    matchup_dataframe: pandas.DataFrame,
    variable_name: str,
) -> pandas.DataFrame:
    depth_bins = DEPTH_BINS_BY_VARIABLE.get(variable_name, DEPTH_BINS_DEFAULT)
    results = []

    for lead_index in sorted(matchup_dataframe["lead_day"].unique()):
        lead_data = matchup_dataframe[matchup_dataframe["lead_day"] == lead_index]
        results.extend(_compute_rmsd_for_all_depth_bins(lead_data, variable_name, lead_index, depth_bins))

    return pandas.DataFrame(results)


def _compute_rmsd_for_all_depth_bins(
    lead_data: pandas.DataFrame,
    variable_name: str,
    lead_index: int,
    depth_bins: dict,
) -> list[dict]:
    results = []

    for depth_label, (minimum_depth, maximum_depth) in depth_bins.items():
        depth_data = lead_data[(lead_data["depth"] >= minimum_depth) & (lead_data["depth"] < maximum_depth)]

        if len(depth_data) > 0:
            rmsd = numpy.sqrt(numpy.mean((depth_data["model_value"] - depth_data[variable_name]) ** 2))
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


def _format_rmsd_table(
    combined: pandas.DataFrame,
    selected_lead_days: list = None,
) -> pandas.DataFrame:
    if selected_lead_days is None:
        selected_lead_days = SELECTED_LEAD_DAYS_FOR_DISPLAY

    filtered_table = combined[combined["lead_day"].isin(selected_lead_days)]

    if filtered_table.empty:
        return pandas.DataFrame()

    pivot = _create_pivot_table(filtered_table)
    pivot = _add_observation_counts(pivot, filtered_table, selected_lead_days[0])
    pivot = _apply_display_names_and_sorting(pivot)

    return _create_final_output_table(pivot, selected_lead_days)


def _create_pivot_table(table: pandas.DataFrame) -> pandas.DataFrame:
    return table.pivot_table(
        values="rmsd",
        index=["variable", "depth_bin"],
        columns="lead_day",
        aggfunc="first",
    ).reset_index()


def _add_observation_counts(
    pivot: pandas.DataFrame,
    table: pandas.DataFrame,
    reference_lead: int,
) -> pandas.DataFrame:
    counts = (
        table[table["lead_day"] == reference_lead]
        .pivot_table(values="count", index=["variable", "depth_bin"], aggfunc="first")
        .reset_index()
    )
    return pivot.merge(counts, on=["variable", "depth_bin"], how="left")


def _apply_display_names_and_sorting(pivot: pandas.DataFrame) -> pandas.DataFrame:
    pivot["variable"] = pivot["variable"].map(VARIABLE_DISPLAY_NAMES)
    pivot["variable_sort_order"] = pivot["variable"].map(
        {name: index for index, name in enumerate(VARIABLE_DISPLAY_ORDER)}
    )
    pivot["depth_sort_order"] = pivot["depth_bin"].map(
        {name: index for index, name in enumerate(DEPTH_BIN_DISPLAY_ORDER)}
    )
    return pivot.sort_values(["variable_sort_order", "depth_sort_order"]).reset_index(drop=True)


def _create_final_output_table(
    pivot: pandas.DataFrame,
    selected_lead_days: list,
) -> pandas.DataFrame:
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
