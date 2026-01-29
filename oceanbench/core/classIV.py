# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray
import pandas
import time
from oceanbench.core.dataset_utils import Dimension, Variable


def perform_matchup(challenger: xarray.Dataset, obs_df: pandas.DataFrame, var_name: str) -> pandas.DataFrame:
    results = []
    run_dates = challenger[Dimension.FIRST_DAY_DATETIME.key()].values

    for run_date in run_dates:
        forecast = challenger.sel({Dimension.FIRST_DAY_DATETIME.key(): run_date})

        for lead in range(10):
            daily_matchup = _match_single_lead_day(forecast, obs_df, var_name, lead, run_date)
            if daily_matchup is not None:
                results.append(daily_matchup)

    return pandas.concat(results, ignore_index=True) if results else pandas.DataFrame()


def _match_single_lead_day(
    forecast: xarray.Dataset, obs_df: pandas.DataFrame, var_name: str, lead: int, run_date
) -> pandas.DataFrame:
    valid_time = pandas.to_datetime(run_date) + pandas.Timedelta(days=lead + 1)
    daily_obs = obs_df[obs_df["time"].dt.date == valid_time.date()].copy()

    if daily_obs.empty:
        return None

    try:
        model_slice = forecast.sel({Dimension.LEAD_DAY_INDEX.key(): lead}).load()
        daily_obs["model_val"] = _interpolate_model_to_observations(model_slice, daily_obs, var_name)
        daily_obs["lead_day"] = lead
        daily_obs["run_date"] = run_date

        return daily_obs.dropna(subset=["model_val", var_name])
    except Exception:
        return None


def _interpolate_model_to_observations(
    model_slice: xarray.Dataset, obs_df: pandas.DataFrame, var_name: str
) -> numpy.ndarray:
    horizontal_coords = {
        "lat": xarray.DataArray(obs_df["lat"].values, dims="pts"),
        "lon": xarray.DataArray(obs_df["lon"].values, dims="pts"),
    }
    horizontal_interp = model_slice[var_name].interp(horizontal_coords, method="linear")

    if _has_vertical_dimension(model_slice[var_name]) and "depth" in obs_df.columns:
        return _interpolate_vertical(horizontal_interp, obs_df["depth"].values)

    return horizontal_interp.values


def _has_vertical_dimension(data_array: xarray.DataArray) -> bool:
    return "depth" in data_array.dims


def _interpolate_vertical(horizontal_interp: xarray.DataArray, depths: numpy.ndarray) -> numpy.ndarray:
    model_values = [_interpolate_single_depth(horizontal_interp, idx, depth) for idx, depth in enumerate(depths)]
    return pandas.to_numeric(model_values, errors="coerce")


def _interpolate_single_depth(horizontal_interp: xarray.DataArray, idx: int, obs_depth: float) -> float:
    try:
        result = horizontal_interp.isel(pts=idx).interp(depth=obs_depth, method="cubic").values
        return float(result)
    except (ValueError, KeyError):
        return numpy.nan


def rmsd_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list,
    depth_bins_config: dict = None,
) -> pandas.DataFrame:

    print("\n" + "=" * 70)
    print("CLASS-4 VALIDATION - TIMING REPORT")
    print("=" * 70)
    total_start = time.time()

    all_results = []

    for var in variables:
        var_name = _get_standard_variable_name(var)
        print(f"\n Processing {var_name}...")

        if var_name not in reference_dataset:
            print("   Variable not in reference dataset, skipping")
            continue

        df_start = time.time()
        obs_df = _create_observation_dataframe(reference_dataset, var_name)
        print(f"  ✓ DataFrame created in {time.time() - df_start:.1f}s ({len(obs_df)} obs)")

        if obs_df.empty:
            print("   No valid observations, skipping")
            continue

        matchup_df = perform_matchup(challenger_dataset, obs_df, var_name)

        if not matchup_df.empty:
            rmsd_start = time.time()
            rmsd_result = rmsd_class4(matchup_df, var_name)
            print(f"  ✓ RMSD computed in {time.time() - rmsd_start:.1f}s")
            all_results.append(rmsd_result)
        else:
            print("   No matchups found")

    if not all_results:
        print("\n No results generated")
        return pandas.DataFrame()

    format_start = time.time()
    result = _format_rmsd_table(pandas.concat(all_results, ignore_index=True))
    print(f"\n Table formatted in {time.time() - format_start:.1f}s")

    total_time = time.time() - total_start
    print("=" * 70)
    print(f"TOTAL TIME: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 70 + "\n")

    return result


def _get_standard_variable_name(var: Variable) -> str:
    short_names = {
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID: "zos",
        Variable.SEA_WATER_POTENTIAL_TEMPERATURE: "thetao",
        Variable.SEA_WATER_SALINITY: "so",
        Variable.EASTWARD_SEA_WATER_VELOCITY: "uo",
        Variable.NORTHWARD_SEA_WATER_VELOCITY: "vo",
    }
    return short_names[var]


def _create_observation_dataframe(reference_dataset: xarray.Dataset, var_name: str) -> pandas.DataFrame:
    obs_df = pandas.DataFrame(
        {
            var_name: reference_dataset[var_name].values,
            "time": reference_dataset["time"].values,
            "lat": reference_dataset["latitude"].values,
            "lon": reference_dataset["longitude"].values,
            "depth": reference_dataset["depth"].values,
            "first_day_datetime": reference_dataset["first_day_datetime"].values,
        }
    )
    return obs_df.dropna(subset=[var_name])


def rmsd_class4(matchup_df: pandas.DataFrame, var_name: str) -> pandas.DataFrame:
    depth_bins = _get_depth_bins_for_variable(var_name)
    results = []

    for lead in sorted(matchup_df["lead_day"].unique()):
        lead_data = matchup_df[matchup_df["lead_day"] == lead]
        results.extend(_compute_rmsd_for_depth_bins(lead_data, var_name, lead, depth_bins))

    return pandas.DataFrame(results)


def _get_depth_bins_for_variable(var_name: str) -> dict:
    depth_bins_by_var = {
        "uo": {"15m": (10, 20)},
        "vo": {"15m": (10, 20)},
        "zos": {"surface": (-1, 1)},
    }

    default_bins = {
        "0-5m": (0, 5),
        "5-100m": (5, 100),
        "100-300m": (100, 300),
        "300-600m": (300, 600),
    }

    return depth_bins_by_var.get(var_name, default_bins)


def _compute_rmsd_for_depth_bins(lead_data: pandas.DataFrame, var_name: str, lead: int, depth_bins: dict) -> list:
    results = []

    for depth_label, (min_depth, max_depth) in depth_bins.items():
        depth_data = _filter_by_depth_range(lead_data, min_depth, max_depth)

        if len(depth_data) > 0:
            rmsd = _calculate_rmsd(depth_data["model_val"], depth_data[var_name])
            results.append(
                {
                    "variable": var_name,
                    "depth_bin": depth_label,
                    "lead_day": lead,
                    "rmsd": rmsd,
                    "count": len(depth_data),
                }
            )

    return results


def _filter_by_depth_range(data: pandas.DataFrame, min_depth: float, max_depth: float) -> pandas.DataFrame:
    return data[(data["depth"] >= min_depth) & (data["depth"] < max_depth)]


def _calculate_rmsd(model_values: numpy.ndarray, obs_values: numpy.ndarray) -> float:
    return numpy.sqrt(numpy.mean((model_values - obs_values) ** 2))


def _create_pivot_table_multi_lead(table: pandas.DataFrame) -> pandas.DataFrame:
    """Create pivot table with lead days as columns."""
    return table.pivot_table(
        values="rmsd", index=["variable", "depth_bin"], columns="lead_day", aggfunc="first"
    ).reset_index()


def _format_rmsd_table(combined: pandas.DataFrame, lead_days: list = None) -> pandas.DataFrame:
    """Format RMSD results with multiple lead days."""
    if lead_days is None:
        lead_days = [0, 2, 4, 6, 9]

    table = combined[combined["lead_day"].isin(lead_days)]

    if table.empty:
        return pandas.DataFrame()

    pivot = _create_pivot_table_multi_lead(table)
    pivot = _add_observation_counts(pivot, table, lead_days[0])
    pivot = _rename_and_sort_variables(pivot)

    return _create_final_output_table_multi_lead(pivot, lead_days)


def _create_final_output_table_multi_lead(pivot: pandas.DataFrame, lead_days: list) -> pandas.DataFrame:
    """Create final output table with multiple lead day columns."""
    output_cols = {
        "Variable": pivot["variable"],
        "Depth Range": pivot["depth_bin"],
    }

    # Mapping: index interne → label affiché
    lead_day_labels_map = {0: 1, 2: 3, 4: 5, 6: 7, 9: 10}

    for lead in lead_days:
        if lead in pivot.columns:
            display_lead = lead_day_labels_map.get(lead, lead + 1)
            output_cols[f"Lead {display_lead}d"] = pivot[lead].apply(lambda x: f"{x:.3f}" if pandas.notna(x) else "-")

    if "count" in pivot.columns:
        output_cols["N obs"] = pivot["count"].apply(lambda x: int(x) if pandas.notna(x) else "-")

    return pandas.DataFrame(output_cols)


def _add_observation_counts(pivot: pandas.DataFrame, table: pandas.DataFrame, reference_lead: int) -> pandas.DataFrame:
    """Add observation counts from reference lead day."""
    counts = (
        table[table["lead_day"] == reference_lead]
        .pivot_table(values="count", index=["variable", "depth_bin"], aggfunc="first")
        .reset_index()
    )

    return pivot.merge(counts, on=["variable", "depth_bin"], how="left")


def _rename_and_sort_variables(pivot: pandas.DataFrame) -> pandas.DataFrame:
    """Rename variables to display names and sort."""
    var_names = {
        "zos": "SSH",
        "thetao": "Temperature",
        "so": "Salinity",
        "uo": "Zonal current",
        "vo": "Meridional current",
    }
    var_order = ["Temperature", "Salinity", "SSH", "Zonal current", "Meridional current"]
    depth_order = ["surface", "0-5m", "5-100m", "100-300m", "300-600m", "15m"]

    pivot["variable"] = pivot["variable"].map(var_names)
    pivot["var_sort"] = pivot["variable"].map({v: i for i, v in enumerate(var_order)})
    pivot["depth_sort"] = pivot["depth_bin"].map({d: i for i, d in enumerate(depth_order)})

    return pivot.sort_values(["var_sort", "depth_sort"]).reset_index(drop=True)
