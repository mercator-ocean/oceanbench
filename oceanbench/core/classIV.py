# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy as np
import pandas as pd
import xarray as xr
from oceanbench.core.dataset_utils import Dimension


def perform_matchup(challenger: xr.Dataset, obs_df: pd.DataFrame, var_name: str) -> pd.DataFrame:
    """Match model values to observation points via bilinear interpolation."""
    results = []
    run_dates = challenger[Dimension.FIRST_DAY_DATETIME.key()].values
    
    for run_date in run_dates:
        forecast = challenger.sel({Dimension.FIRST_DAY_DATETIME.key(): run_date})
        
        for lead in range(10):
            valid_time = pd.to_datetime(run_date) + pd.Timedelta(days=lead)
            
            daily_obs = obs_df[
                (obs_df['time'].dt.date == valid_time.date())
            ].copy()
            
            if daily_obs.empty:
                continue
            
            model_slice = forecast.sel({Dimension.LEAD_DAY_INDEX.key(): lead})
            
            coords = {
                "lat": xr.DataArray(daily_obs['lat'].values, dims="pts"),
                "lon": xr.DataArray(daily_obs['lon'].values, dims="pts")
            }
            
            if "depth" in model_slice.dims and "depth" in daily_obs.columns:
                coords["depth"] = xr.DataArray(daily_obs['depth'].values, dims="pts")
            
            try:
                daily_obs['model_val'] = model_slice[var_name].interp(coords, method="linear").values
                daily_obs['lead_day'] = lead
                daily_obs = daily_obs.dropna(subset=['model_val', var_name])
                results.append(daily_obs)
            except Exception as e:
                continue
    
    return pd.concat(results) if results else pd.DataFrame()


def compute_metrics(matchup_df: pd.DataFrame, var_name: str, clim_df: pd.DataFrame = None) -> pd.DataFrame:
    """Compute RMSE, Bias, ACC per lead day."""
    
    def stats_per_lead(group):
        m = group['model_val']
        o = group[var_name]
        
        if clim_df is not None and 'clim_val' in group.columns:
            m_ano = m - group['clim_val']
            o_ano = o - group['clim_val']
            acc = m_ano.corr(o_ano)
        else:
            acc = m.corr(o)
        
        return pd.Series({
            'rmse': np.sqrt(((m - o)**2).mean()),
            'bias': (m - o).mean(),
            'acc': acc,
            'count': len(group)
        })
    
    return matchup_df.groupby('lead_day').apply(stats_per_lead)
