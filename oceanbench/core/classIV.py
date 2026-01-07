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
    - Cubic spline vertical (depth)
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

                if "depth" in model_slice.dims and "depth" in daily_obs.columns:
                    model_values = [
                        _interpolate_depth(horizontal_interp, idx, obs_depth)
                        for idx, obs_depth in enumerate(daily_obs["depth"].values)
                    ]
                    daily_obs["model_val"] = model_values
                else:
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
    Interpolate depth with fallback from cubic to linear.
    Returns np.nan if both methods fail.
    """
    point_data = horizontal_interp.isel(pts=idx)

    # Try cubic interpolation first
    try:
        return point_data.interp(depth=obs_depth, method="cubic").values
    except (ValueError, KeyError):
        # Fallback to linear interpolation
        try:
            return point_data.interp(depth=obs_depth, method="linear").values
        except (ValueError, KeyError):
            return np.nan


def rmsd_class4(matchup_df: pd.DataFrame, var_name: str) -> pd.DataFrame:
    """
    Compute Root Mean Square Difference for Class 4 validation.
    parameters:
    -----------
    matchup_df : pd.DataFrame
        DataFrame with model-observation pairs from perform_matchup()
        Must contain columns: 'model_val', var_name, 'lead_day'
    var_name : str
        Name of the observed variable column

    returns:
    --------
    pd.DataFrame
        RMSD and count per lead_day
    """
    rmsd_values = []
    counts = []
    lead_days = []

    for lead in sorted(matchup_df["lead_day"].unique()):
        group = matchup_df[matchup_df["lead_day"] == lead]
        model_vals = group["model_val"].values
        obs_vals = group[var_name].values

        rmsd = np.sqrt(np.mean((model_vals - obs_vals) ** 2))

        rmsd_values.append(rmsd)
        counts.append(len(group))
        lead_days.append(lead)

    return pd.DataFrame({"lead_day": lead_days, "rmsd": rmsd_values, "count": counts}).set_index("lead_day")
