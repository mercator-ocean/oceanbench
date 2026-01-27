# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy as np
import pandas as pd
import xarray as xr
from oceanbench.core.dataset_utils import Dimension


def perform_matchup(challenger: xr.Dataset, obs_df: pd.DataFrame, var_name: str) -> pd.DataFrame:
    """
    Match model values to observation points.
    Interpolation according to the Class-4 method :
    - Horizontal bilinear (lat/lon)
    - Cubic spline vertical (depth) - only for 3D variables
    """
    results = []
    run_dates = challenger[Dimension.FIRST_DAY_DATETIME.key()].values

    for run_date in run_dates:
        forecast = challenger.sel({Dimension.FIRST_DAY_DATETIME.key(): run_date})

        for lead in range(10):
            valid_time = pd.to_datetime(run_date) + pd.Timedelta(days=lead)
            daily_obs = obs_df[(obs_df["time"].dt.date == valid_time.date())].copy()

            if daily_obs.empty:
                continue

            try:
                model_slice = forecast.sel({Dimension.LEAD_DAY_INDEX.key(): lead}).load()

                coords_horizontal = {
                    "lat": xr.DataArray(daily_obs["lat"].values, dims="pts"),
                    "lon": xr.DataArray(daily_obs["lon"].values, dims="pts"),
                }

                horizontal_interp = model_slice[var_name].interp(coords_horizontal, method="linear")

                # Vérifier si la VARIABLE a une dimension depth
                if "depth" in model_slice[var_name].dims and "depth" in daily_obs.columns:
                    # Variable 3D - interpoler verticalement
                    model_values = [
                        _interpolate_depth(horizontal_interp, idx, obs_depth)
                        for idx, obs_depth in enumerate(daily_obs["depth"].values)
                    ]
                    daily_obs["model_val"] = model_values
                    # Convertir en float pour éviter dtype object
                    daily_obs["model_val"] = pd.to_numeric(daily_obs["model_val"], errors="coerce")
                else:
                    # Variable 2D (comme zos) - juste interpolation horizontale
                    daily_obs["model_val"] = horizontal_interp.values

                daily_obs["lead_day"] = lead
                daily_obs["run_date"] = run_date
                daily_obs = daily_obs.dropna(subset=["model_val", var_name])

                if not daily_obs.empty:
                    results.append(daily_obs)

            except Exception as e:
                print(f"⚠️ Erreur pour {run_date} lead={lead}: {e}")
                continue

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def _interpolate_depth(horizontal_interp: xr.DataArray, idx: int, obs_depth: float) -> float:
    """
    Interpolate depth with LINEAR method (more robust than cubic).
    """
    point_data = horizontal_interp.isel(pts=idx)

    try:
        result = point_data.interp(depth=obs_depth, method="linear").values
        return float(result)  # Forcer en float
    except (ValueError, KeyError):
        return np.nan


def rmsd_class4(matchup_df: pd.DataFrame, var_name: str) -> pd.DataFrame:
    """
    Compute RMSD by lead_day AND depth bins.

    Parameters:
    -----------
    matchup_df : pd.DataFrame
        DataFrame with model-observation pairs from perform_matchup()
    var_name : str
        Name of the variable

    Returns:
    --------
    pd.DataFrame
        RMSD and count per lead_day and depth_bin
    """
    # Définir les bins de profondeur selon la variable
    if var_name in ["uo", "vo"]:
        # Pour les courants: seulement 15m (±5m autour de 15m)
        depth_bins = {"15m": (10, 20)}
    elif var_name == "zos":
        # Pour SSH: surface
        depth_bins = {"surface": (-1, 1)}
    else:
        # Pour température et salinité: bins standards
        depth_bins = {
            "0-5m": (0, 5),
            "5-100m": (5, 100),
            "100-300m": (100, 300),
            "300-600m": (300, 600),
        }

    results = []

    for lead in sorted(matchup_df["lead_day"].unique()):
        lead_data = matchup_df[matchup_df["lead_day"] == lead]

        # RMSD par bin de profondeur
        for depth_label, (min_depth, max_depth) in depth_bins.items():
            depth_data = lead_data[(lead_data["depth"] >= min_depth) & (lead_data["depth"] < max_depth)]

            if len(depth_data) > 0:
                model_vals = depth_data["model_val"].values
                obs_vals = depth_data[var_name].values
                rmsd = np.sqrt(np.mean((model_vals - obs_vals) ** 2))

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
