# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy as np
import pandas as pd
import xarray as xr
import xarray
import pandas
from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.dataset_utils import Variable


def perform_matchup(challenger: xr.Dataset, obs_df: pd.DataFrame, var_name: str) -> pd.DataFrame:
    results = []
    run_dates = challenger[Dimension.FIRST_DAY_DATETIME.key()].values

    for run_date in run_dates:
        forecast = challenger.sel({Dimension.FIRST_DAY_DATETIME.key(): run_date})

        for lead in range(10):
            valid_time = pd.to_datetime(run_date) + pd.Timedelta(days=lead)
            daily_obs = obs_df[obs_df["time"].dt.date == valid_time.date()].copy()

            if daily_obs.empty:
                continue

            try:
                model_slice = forecast.sel({Dimension.LEAD_DAY_INDEX.key(): lead}).load()

                # Horizontal interpolation
                coords_horizontal = {
                    "lat": xr.DataArray(daily_obs["lat"].values, dims="pts"),
                    "lon": xr.DataArray(daily_obs["lon"].values, dims="pts"),
                }
                horizontal_interp = model_slice[var_name].interp(coords_horizontal, method="linear")

                # Vertical interpolation if needed
                if "depth" in model_slice[var_name].dims and "depth" in daily_obs.columns:
                    model_values = [
                        _interpolate_depth(horizontal_interp, idx, obs_depth)
                        for idx, obs_depth in enumerate(daily_obs["depth"].values)
                    ]
                    daily_obs["model_val"] = pd.to_numeric(model_values, errors="coerce")
                else:
                    daily_obs["model_val"] = horizontal_interp.values

                daily_obs["lead_day"] = lead
                daily_obs["run_date"] = run_date
                daily_obs = daily_obs.dropna(subset=["model_val", var_name])

                if not daily_obs.empty:
                    results.append(daily_obs)

            except Exception:
                continue

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def _interpolate_depth(horizontal_interp: xr.DataArray, idx: int, obs_depth: float) -> float:
    """Interpolate depth using cubic method."""
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

    # Default variable name mapping (challenger → reference)
    var_mapping = {
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID: ("zos", "SLEV"),
        Variable.SEA_WATER_POTENTIAL_TEMPERATURE: ("thetao", "TEMP"),
        Variable.SEA_WATER_SALINITY: ("so", "PSAL"),
        Variable.EASTWARD_SEA_WATER_VELOCITY: ("uo", "EWCT"),
        Variable.NORTHWARD_SEA_WATER_VELOCITY: ("vo", "NSCT"),
    }

    all_results = []

    for var in variables:
        challenger_var_name, obs_var_name = var_mapping[var]

        if obs_var_name not in reference_dataset:
            continue

        # Create observation DataFrame
        obs_df = pd.DataFrame(
            {
                challenger_var_name: reference_dataset[obs_var_name].values,
                "time": reference_dataset["time"].values,
                "lat": reference_dataset["latitude"].values,
                "lon": reference_dataset["longitude"].values,
                "depth": reference_dataset["depth"].values,
                "first_day_datetime": reference_dataset["first_day_datetime"].values,
            }
        ).dropna(subset=[challenger_var_name])

        if obs_df.empty:
            continue

        # Matchup and compute RMSD
        matchup_df = perform_matchup(challenger_dataset, obs_df, challenger_var_name)
        if not matchup_df.empty:
            all_results.append(rmsd_class4(matchup_df, challenger_var_name))

    if not all_results:
        return pd.DataFrame()

    return _format_rmsd_table(pd.concat(all_results, ignore_index=True))


def _format_rmsd_table(combined: pd.DataFrame, lead_day: int = 1) -> pd.DataFrame:
    table = combined[combined["lead_day"] == lead_day]

    pivot = table.pivot_table(values=["rmsd", "count"], index=["variable", "depth_bin"], aggfunc="first").reset_index()

    # Rename and sort
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
    pivot = pivot.sort_values(["var_sort", "depth_sort"]).reset_index(drop=True)

    return pd.DataFrame(
        {
            "Variable": pivot["variable"],
            "Depth Range": pivot["depth_bin"],
            f"RMSE (lead={lead_day} day)": pivot["rmsd"].apply(lambda x: f"{x:.3f}"),
            "N observations": pivot["count"].astype(int),
        }
    )


def rmsd_class4(matchup_df: pd.DataFrame, var_name: str) -> pd.DataFrame:
    """Compute RMSD by lead_day and depth bins."""

    depth_bins = {
        "uo": {"15m": (10, 20)},
        "vo": {"15m": (10, 20)},
        "zos": {"surface": (-1, 1)},
    }.get(
        var_name,
        {
            "0-5m": (0, 5),
            "5-100m": (5, 100),
            "100-300m": (100, 300),
            "300-600m": (300, 600),
        },
    )

    results = []
    for lead in sorted(matchup_df["lead_day"].unique()):
        lead_data = matchup_df[matchup_df["lead_day"] == lead]

        for depth_label, (min_depth, max_depth) in depth_bins.items():
            depth_data = lead_data[(lead_data["depth"] >= min_depth) & (lead_data["depth"] < max_depth)]

            if len(depth_data) > 0:
                rmsd = np.sqrt(np.mean((depth_data["model_val"] - depth_data[var_name]) ** 2))
                results.append(
                    {
                        "variable": var_name,
                        "depth_bin": depth_label,
                        "lead_day": lead,
                        "rmsd": rmsd,
                        "count": len(depth_data),
                    }
                )

    return pd.DataFrame(results)
