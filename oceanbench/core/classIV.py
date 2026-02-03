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
REANALYSIS_MSSH_SHIFT = -0.1148
MSSH_URL = "https://minio.dive.edito.eu/project-ml-compression/public/glorys12_mssh_2024.zarr"

VARIABLE_DISPLAY_NAMES: dict[str, str] = {
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): "SSH",
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): "Temperature",
    Variable.SEA_WATER_SALINITY.key(): "Salinity",
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(): "Zonal current",
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): "Meridional current",
}

_MSSH_CACHE = None  # Cache global


def _load_mean_ssh() -> xarray.DataArray:
    """Load pre-computed MSSH from GLORYS12 2024 (cached)."""
    global _MSSH_CACHE

    if _MSSH_CACHE is not None:
        print("✓ Using cached MSSH from memory", flush=True)
        return _MSSH_CACHE

    print(f"📥 Loading MSSH from {MSSH_URL}...", flush=True)
    mssh_dataset = xarray.open_dataset(MSSH_URL, engine="zarr", chunks=None)

    print("   Computing MSSH in memory...", flush=True)
    _MSSH_CACHE = mssh_dataset["mssh"].load()  # Force en mémoire

    print(f"✓ MSSH loaded in memory ({_MSSH_CACHE.nbytes / 1e6:.1f} MB)", flush=True)

    return _MSSH_CACHE


def _convert_sla_to_ssh(reference_dataset: xarray.Dataset, mean_ssh: xarray.DataArray) -> xarray.Dataset:
    """Convert SLA observations to absolute SSH by adding MSSH and correction shift."""
    ssh_variable = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()

    if ssh_variable not in reference_dataset:
        return reference_dataset

    print(" Converting SLA to absolute SSH...", flush=True)
    print(f"   Number of observations: {len(reference_dataset.obs)}", flush=True)

    # Interpolate MSSH to observation locations
    print("   Interpolating MSSH to observation locations...", flush=True)
    mssh_at_obs = mean_ssh.interp(
        {
            Dimension.LATITUDE.key(): reference_dataset[Dimension.LATITUDE.key()],
            Dimension.LONGITUDE.key(): reference_dataset[Dimension.LONGITUDE.key()],
        },
        method="linear",
    ).compute()  # Force le calcul

    print("   Computing SSH = SLA + MSSH + shift...", flush=True)
    # Convert: ssh_absolute = sla + mssh + shift
    reference_dataset[ssh_variable] = reference_dataset[ssh_variable] + mssh_at_obs + REANALYSIS_MSSH_SHIFT

    print(" SLA converted to SSH", flush=True)
    return reference_dataset


def _standardize_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    """Normalize dimension and variable names while preserving non-standard coordinates."""
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
    print(f"         Starting matchup for {len(observations_dataframe)} obs...", flush=True)
    matchup_results = []
    run_dates = challenger[Dimension.FIRST_DAY_DATETIME.key()].values

    print(f"         Number of run_dates: {len(run_dates)}", flush=True)
    print(f"         LEAD_DAYS_COUNT: {LEAD_DAYS_COUNT}", flush=True)

    for i, run_date in enumerate(run_dates):
        print(f"Run date {i + 1}/{len(run_dates)}: {run_date}", flush=True)
        forecast = challenger.sel({Dimension.FIRST_DAY_DATETIME.key(): run_date})

        for lead_index in range(LEAD_DAYS_COUNT):
            print(f"            Lead day {lead_index}...", flush=True)
            daily_matchup = _match_single_lead_day(
                forecast, observations_dataframe, variable_name, lead_index, run_date
            )
            if daily_matchup is not None:
                print(f"            → {len(daily_matchup)} matchups found", flush=True)
                matchup_results.append(daily_matchup)
            else:
                print("No matchups", flush=True)

    total_matchups = sum(len(r) for r in matchup_results)
    print(f"         Matchup complete: {total_matchups} total", flush=True)
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

    model_slice = forecast.sel({Dimension.LEAD_DAY_INDEX.key(): lead_index})
    daily_observations["model_value"] = _interpolate_model_to_observations(
        model_slice, daily_observations, variable_name
    )
    daily_observations["lead_day"] = lead_index
    daily_observations["run_date"] = run_date

    return daily_observations.dropna(subset=["model_value", variable_name])


def _interpolate_model_to_observations(
    model_slice: xarray.Dataset, observations_dataframe: pandas.DataFrame, variable_name: str
) -> numpy.ndarray:

    data_array = model_slice[variable_name]

    # Créer les coordonnées pour interpolation
    lats = xarray.DataArray(observations_dataframe[Dimension.LATITUDE.key()].values, dims="points")
    lons = xarray.DataArray(observations_dataframe[Dimension.LONGITUDE.key()].values, dims="points")

    # Interpolation horizontale
    horizontal_interp = data_array.interp(
        {Dimension.LATITUDE.key(): lats, Dimension.LONGITUDE.key(): lons}, method="linear"
    )

    # Si pas de dimension verticale, on retourne directement
    if Dimension.DEPTH.key() not in data_array.dims:
        return horizontal_interp.values

    # Interpolation verticale VECTORISÉE
    if Dimension.DEPTH.key() in observations_dataframe.columns:
        depths = xarray.DataArray(observations_dataframe[Dimension.DEPTH.key()].values, dims="points")
        # Interpolation 3D en une seule opération
        result = horizontal_interp.interp({Dimension.DEPTH.key(): depths}, method="linear")
        return result.values

    return horizontal_interp.values


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
            .interp({Dimension.DEPTH.key(): observation_depth}, method="linear")
            .values
        )
        return float(result) if not numpy.isnan(result) else numpy.nan
    except (ValueError, KeyError):
        return numpy.nan


def rmsd_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
    use_ssh_correction: bool = True,
) -> pandas.DataFrame:
    """
    Compute Class-4 validation metrics.

    Args:
        challenger_dataset: Forecast dataset to validate
        reference_dataset: In-situ observations dataset
        variables: List of variables to validate
        use_ssh_correction: Apply SLA to SSH conversion using GLORYS MSSH (default: True)
    """
    print(" Starting Class-4 validation...", flush=True)

    print("   Standardizing challenger...", flush=True)
    challenger_standardized = _standardize_dataset(challenger_dataset)
    print("    Challenger standardized", flush=True)

    print("   Standardizing reference...", flush=True)
    reference_standardized = _standardize_dataset(reference_dataset)
    print("    Reference standardized", flush=True)

    # Convert SLA to absolute SSH if requested
    ssh_variable = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
    if use_ssh_correction and ssh_variable in reference_standardized:
        print("   Loading MSSH...", flush=True)
        mean_ssh = _load_mean_ssh()
        reference_standardized = _convert_sla_to_ssh(reference_standardized, mean_ssh)

    print("   Starting variable processing...", flush=True)
    all_results = []

    for variable in variables:
        variable_name = variable.key()
        print(f"\n   Processing {variable_name}...", flush=True)

        if variable_name not in reference_standardized:
            print("  Not in reference, skipping", flush=True)
            continue

        observations_dataframe = _create_observation_dataframe(reference_standardized, variable_name)
        print(f"      ✓ {len(observations_dataframe)} observations", flush=True)

        if observations_dataframe.empty:
            continue

        matchup_dataframe = perform_matchup(challenger_standardized, observations_dataframe, variable_name)
        print(f"      ✓ {len(matchup_dataframe)} matchups", flush=True)

        if not matchup_dataframe.empty:
            print("Computing RMSD...", flush=True)
            rmsd_result = rmsd_class4(matchup_dataframe, variable_name)
            print("RMSD computed", flush=True)
            all_results.append(rmsd_result)

    if not all_results:
        print("\n No results generated", flush=True)
        return pandas.DataFrame()

    print("\n   Formatting results...", flush=True)
    result = _format_rmsd_table(pandas.concat(all_results, ignore_index=True))
    print("✓ Class-4 validation complete!", flush=True)

    return result


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
    print("RMSD calculation starting...", flush=True)
    depth_bins = _get_depth_bins_for_variable(variable_name)
    results = []

    for lead_index in sorted(matchup_dataframe["lead_day"].unique()):
        lead_data = matchup_dataframe[matchup_dataframe["lead_day"] == lead_index]
        results.extend(_compute_rmsd_for_depth_bins(lead_data, variable_name, lead_index, depth_bins))

    print("RMSD calculation done", flush=True)
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
