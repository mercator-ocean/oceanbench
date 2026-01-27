# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy as np
import pandas as pd
import xarray as xr
import xarray
import pandas
from oceanbench.core.dataset_utils import Dimension, Variable


def perform_matchup(challenger: xr.Dataset, obs_df: pd.DataFrame, var_name: str) -> pd.DataFrame:
    results = []
    run_dates = challenger[Dimension.FIRST_DAY_DATETIME.key()].values

    for run_date in run_dates:
        forecast = challenger.sel({Dimension.FIRST_DAY_DATETIME.key(): run_date})

        for lead in range(10):
            daily_matchup = _match_single_lead_day(forecast, obs_df, var_name, lead, run_date)
            if daily_matchup is not None:
                results.append(daily_matchup)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def _match_single_lead_day(
    forecast: xr.Dataset, obs_df: pd.DataFrame, var_name: str, lead: int, run_date
) -> pd.DataFrame:
    valid_time = pd.to_datetime(run_date) + pd.Timedelta(days=lead)
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


def _interpolate_model_to_observations(model_slice: xr.Dataset, obs_df: pd.DataFrame, var_name: str) -> np.ndarray:
    horizontal_coords = {
        "lat": xr.DataArray(obs_df["lat"].values, dims="pts"),
        "lon": xr.DataArray(obs_df["lon"].values, dims="pts"),
    }
    horizontal_interp = model_slice[var_name].interp(horizontal_coords, method="linear")

    if _has_vertical_dimension(model_slice[var_name]) and "depth" in obs_df.columns:
        return _interpolate_vertical(horizontal_interp, obs_df["depth"].values)

    return horizontal_interp.values


def _has_vertical_dimension(data_array: xr.DataArray) -> bool:
    return "depth" in data_array.dims


def _interpolate_vertical(horizontal_interp: xr.DataArray, depths: np.ndarray) -> np.ndarray:
    model_values = [_interpolate_single_depth(horizontal_interp, idx, depth) for idx, depth in enumerate(depths)]
    return pd.to_numeric(model_values, errors="coerce")


def _interpolate_single_depth(horizontal_interp: xr.DataArray, idx: int, obs_depth: float) -> float:
    try:
        result = horizontal_interp.isel(pts=idx).interp(depth=obs_depth, method="cubic").values
        return float(result)
    except (ValueError, KeyError):
        return np.nan


def rmsd_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list,
    depth_bins_config: dict = None,
) -> pandas.DataFrame:

    all_results = []

    for var in variables:
        var_name = _get_standard_variable_name(var)

        if var_name not in reference_dataset:
            continue

        obs_df = _create_observation_dataframe(reference_dataset, var_name)
        if obs_df.empty:
            continue

        matchup_df = perform_matchup(challenger_dataset, obs_df, var_name)
        if not matchup_df.empty:
            all_results.append(rmsd_class4(matchup_df, var_name))

    if not all_results:
        return pd.DataFrame()

    return _format_rmsd_table(pd.concat(all_results, ignore_index=True))


def _get_standard_variable_name(var: Variable) -> str:
    short_names = {
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID: "zos",
        Variable.SEA_WATER_POTENTIAL_TEMPERATURE: "thetao",
        Variable.SEA_WATER_SALINITY: "so",
        Variable.EASTWARD_SEA_WATER_VELOCITY: "uo",
        Variable.NORTHWARD_SEA_WATER_VELOCITY: "vo",
    }
    return short_names[var]


def _create_observation_dataframe(reference_dataset: xr.Dataset, var_name: str) -> pd.DataFrame:
    obs_df = pd.DataFrame(
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


def rmsd_class4(matchup_df: pd.DataFrame, var_name: str) -> pd.DataFrame:
    depth_bins = _get_depth_bins_for_variable(var_name)
    results = []

    for lead in sorted(matchup_df["lead_day"].unique()):
        lead_data = matchup_df[matchup_df["lead_day"] == lead]
        results.extend(_compute_rmsd_for_depth_bins(lead_data, var_name, lead, depth_bins))

    return pd.DataFrame(results)


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


def _compute_rmsd_for_depth_bins(lead_data: pd.DataFrame, var_name: str, lead: int, depth_bins: dict) -> list:
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


def _filter_by_depth_range(data: pd.DataFrame, min_depth: float, max_depth: float) -> pd.DataFrame:
    return data[(data["depth"] >= min_depth) & (data["depth"] < max_depth)]


def _calculate_rmsd(model_values: np.ndarray, obs_values: np.ndarray) -> float:
    return np.sqrt(np.mean((model_values - obs_values) ** 2))


def _format_rmsd_table(combined: pd.DataFrame, lead_day: int = 1) -> pd.DataFrame:
    table = combined[combined["lead_day"] == lead_day]
    pivot = _create_pivot_table(table)
    pivot = _rename_and_sort_variables(pivot)
    return _create_final_output_table(pivot, lead_day)


def _create_pivot_table(table: pd.DataFrame) -> pd.DataFrame:
    return table.pivot_table(values=["rmsd", "count"], index=["variable", "depth_bin"], aggfunc="first").reset_index()


def _rename_and_sort_variables(pivot: pd.DataFrame) -> pd.DataFrame:
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


def _create_final_output_table(pivot: pd.DataFrame, lead_day: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Variable": pivot["variable"],
            "Depth Range": pivot["depth_bin"],
            f"RMSE (lead={lead_day} day)": pivot["rmsd"].apply(lambda x: f"{x:.3f}"),
            "N observations": pivot["count"].astype(int),
        }
    )
