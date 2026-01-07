# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import logging
import os
import pandas as pd
from xarray import Dataset
import copernicusmarine as cm
from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.climate_forecast_standard_names import StandardVariable

logger = logging.getLogger("copernicusmarine")
logger.setLevel(level=logging.WARNING)


def _get_credentials():
    username = os.getenv("COPERNICUS_USERNAME")
    password = os.getenv("COPERNICUS_PASSWORD")

    if not username or not password:
        logger.error("COPERNICUS_USERNAME and COPERNICUS_PASSWORD not set")
        return None, None

    return username, password


def _login_copernicus():
    username, password = _get_credentials()
    if username and password:
        cm.login(username=username, password=password)


def _argo_profiles(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        _login_copernicus()

        df = cm.read_dataframe(
            dataset_id="cmems_obs-ins_glo_phybgcwav_mynrt_na_irr",
            start_datetime=start_date,
            end_datetime=end_date,
        )

        df = df[df["value_qc"] == 1].copy()

        df = df.pivot_table(
            index=["time", "latitude", "longitude", "depth"], columns="variable", values="value", aggfunc="first"
        ).reset_index()

        df = df.rename(
            columns={
                "time": "time",
                "latitude": "lat",
                "longitude": "lon",
                "depth": "depth",
                "TEMP": StandardVariable.SEA_WATER_POTENTIAL_TEMPERATURE.value,
                "PSAL": StandardVariable.SEA_WATER_SALINITY.value,
            }
        )

        thetao_col = StandardVariable.SEA_WATER_POTENTIAL_TEMPERATURE.value
        so_col = StandardVariable.SEA_WATER_SALINITY.value

        df = df[(df[thetao_col] > -5) & (df[thetao_col] < 50) & (df[so_col] > 30) & (df[so_col] < 40)]

        df["time"] = pd.to_datetime(df["time"])

        logger.info(f"Argo: {len(df)} profiles")
        return df

    except Exception as e:
        logger.error(f"Argo fetch failed: {e}")
        return pd.DataFrame()


def _surface_currents(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        _login_copernicus()

        df = cm.read_dataframe(
            dataset_id="cmems_obs-ins_glo_phy-cur_nrt_drifter_irr",
            start_datetime=start_date,
            end_datetime=end_date,
        )

        df = df[df["value_qc"] == 1].copy()

        df = df.pivot_table(
            index=["time", "latitude", "longitude", "depth"], columns="variable", values="value", aggfunc="first"
        ).reset_index()

        df = df.rename(
            columns={
                "time": "time",
                "latitude": "lat",
                "longitude": "lon",
                "depth": "depth",
                "EWCT": StandardVariable.EASTWARD_SEA_WATER_VELOCITY.value,
                "NSCT": StandardVariable.NORTHWARD_SEA_WATER_VELOCITY.value,
            }
        )

        df["time"] = pd.to_datetime(df["time"])

        logger.info(f"Drifters: {len(df)} measurements")
        return df

    except Exception as e:
        logger.error(f"Drifters fetch failed: {e}")
        return pd.DataFrame()


def _sea_level_anomaly(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        _login_copernicus()

        df = cm.read_dataframe(
            dataset_id="cmems_obs-slv_glo_phy_nrt_008_044",
            variables=["sla"],
            start_datetime=start_date,
            end_datetime=end_date,
        )

        df = df.rename(
            columns={
                "TIME": "time",
                "LATITUDE": "lat",
                "LONGITUDE": "lon",
                "sla": StandardVariable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.value,
            }
        )

        df["time"] = pd.to_datetime(df["time"])

        logger.info(f"SLA: {len(df)} observations")
        return df

    except Exception as e:
        logger.error(f"SLA fetch failed: {e}")
        return pd.DataFrame()


def observations_dataset(challenger_dataset: Dataset) -> dict:
    run_dates = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    start_dt = pd.to_datetime(run_dates.min()).strftime("%Y-%m-%d")
    end_dt = (pd.to_datetime(run_dates.max()) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    logger.info(f"Fetching observations from {start_dt} to {end_dt}")

    obs = {
        "argo": _argo_profiles(start_dt, end_dt),
        "drifters": _surface_currents(start_dt, end_dt),
        "sla": _sea_level_anomaly(start_dt, end_dt),
    }

    return {k: v for k, v in obs.items() if not v.empty}
