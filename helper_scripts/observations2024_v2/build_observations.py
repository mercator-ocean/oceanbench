#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Build the OceanBench class-4 observation store, one zarr per UTC day.

Rewrite of the notebook creation_data_2025.ipynb with a flagged-archive schema:
every row that the source files contain is kept, and rows that fail the default
policy are marked qc_keep=0 with their flags and raw values preserved. The nine
legacy variable names keep their exact legacy dtypes so the existing scorer
reads the new store unchanged; for a policy-failing row the legacy measurement
columns are NaN, so a legacy consumer sees a corrected store.

Credentials are read from the environment only. Nothing is hardcoded.
  COPERNICUSMARINE_SERVICE_USERNAME / COPERNICUSMARINE_SERVICE_PASSWORD
  CF_KEY / CF_SECRET  (falls back to AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)

This script performs no build on import. Run it explicitly.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import gc
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("build_observations")

# ============================================================
# POLICY (edit here, it is the only place decisions live)
# ============================================================

POLICY: dict[str, Any] = {
    "policy_version": "v2.0.0",
    # QC flags accepted for a measurement to reach the legacy columns.
    # OceanSITES reference table 2: 1 = good_data, 2 = probably_good_data.
    # Default is flag 1 only, matching the legacy store. Every row keeps its
    # raw flag, so relaxing to [1, 2] is a scoring-time choice, not a rebuild.
    "accepted_qc_flags": [1],
    # Position and time QC flags accepted for any row.
    "accepted_position_qc_flags": [1, 2],
    "accepted_time_qc_flags": [1, 2],
    # Depth QC accepts 7 = nominal_value in addition to good and probably good.
    # Surface drifters (CO_TS_DB) carry DEPH_QC 7 on every level because their
    # depth is a nominal platform depth rather than a measured one, so [1, 2]
    # alone rejects that entire stream.
    "accepted_depth_qc_flags": [1, 2, 7],
    # Current basis: "filtr" uses EWCT_FILTR / NSCT_FILTR (3-day Lanczos,
    # inertial band removed), "raw" uses EWCT / NSCT.
    "current_basis": "filtr",
    # CURRENT_TEST is a 3-digit SAW drogue-loss code. 011 (int 11) means the
    # drogue is considered missing, so the velocity is surface circulation
    # contaminated by direct wind drag. Those rows are flagged out.
    "drop_undrogued_current_test": [11],
    # Rows with unknown drogue status (CURRENT_TEST absent or fill) are kept
    # when this is True.
    "keep_unknown_drogue": True,
    # Absolute bound on sea level anomaly in metres.
    "sla_abs_max_m": 2.0,
    # Rows whose time falls outside the target UTC day are flagged, not dropped
    # silently and not silently included.
    "enforce_day_alignment": True,
    # Duplicate obs_id within a day: keep the first row that passes policy,
    # else the first row seen.
    "dedup_strategy": "first_kept",
}

# ============================================================
# CONSTANTS
# ============================================================

CLOUDFERRO_ENDPOINT = "https://s3.waw3-1.cloudferro.com"
CLOUDFERRO_REGION = "waw3-1"

SOURCE_BUCKET = "mdl-native-08"
SOURCE_ENDPOINT = CLOUDFERRO_ENDPOINT

CURRENTS_PRODUCT = "INSITU_GLO_PHY_UVASSIM_DISCRETE_NRT_013_054"
CURRENTS_DATASET = "cmems_obs-ins_glo_phy-cur_nrt_drifter-filt-assim_irr_202311"

TS_PRODUCT = "INSITU_GLO_PHY_TSASSIM_DISCRETE_NRT_013_047"
TS_DATASET = "cmems_obs-ins_glo_phy-temp-sal_nrt_assim_irr_202211"

# DUACS L3 my missions, kept deliberately equal to the legacy notebook's set.
# Days of 2024 available per mission, measured 2026-08-05 by listing
# SEALEVEL_GLO_PHY_L3_MY_008_062 with the pattern *_1hz_2024*:
#   alg    366  SARAL/AltiKa drifting phase
#   c2n    365  CryoSat-2 new orbit
#   h2b    281  HY-2B, real gaps inside 2024, absent on 2024-06-15
#   s3a    366
#   s3b    366
#   s6a_lr 366  Sentinel-6A low resolution
#   swon   366  SWOT nadir
#
# "al" (SARAL nominal orbit) replaced by "alg". The legacy notebook asked for
# "al", which then served the drifting-phase data; the catalogue has since split
# the drifting phase into its own dataset and "al" now has zero 2024 files.
#
# j3n (Jason-3 interleaved, 366 days of 2024) is available and deliberately NOT
# included: it was not in the legacy set and adding it inflates 2024-06-15 by
# 50220 points, about 18 percent. Add it only as a conscious basis change.
# Zero 2024 coverage, not candidates: al, c2, en, enn, g2, h2a, h2ag, j3, j3g,
# swonc.
SLA_SATELLITES = {
    "alg": "cmems_obs-sl_glo_phy-ssh_my_alg-l3-duacs_PT1S",
    "c2n": "cmems_obs-sl_glo_phy-ssh_my_c2n-l3-duacs_PT1S",
    "h2b": "cmems_obs-sl_glo_phy-ssh_my_h2b-l3-duacs_PT1S",
    "s3a": "cmems_obs-sl_glo_phy-ssh_my_s3a-l3-duacs_PT1S",
    "s3b": "cmems_obs-sl_glo_phy-ssh_my_s3b-l3-duacs_PT1S",
    "s6a_lr": "cmems_obs-sl_glo_phy-ssh_my_s6a-lr-l3-duacs_PT1S",
    "swon": "cmems_obs-sl_glo_phy-ssh_my_swon-l3-duacs_PT1S",
}

DEFAULT_TARGET = "s3://oceanbench-bucket/dev/observations2024-v2"
DEFAULT_OBS_BASIS_VERSION = "2024-v2.0.1"
DEFAULT_MIN_SATELLITES = 5
RECENT_DAYS = 183  # about 6 months

# Legacy variable names, unchanged.
VAR_DEPTH = "depth"
VAR_LAT = "latitude"
VAR_LON = "longitude"
VAR_TIME = "time"
VAR_ZOS = "sea_surface_height_above_geoid"
VAR_TEMP = "sea_water_potential_temperature"
VAR_SAL = "sea_water_salinity"
VAR_UO = "eastward_sea_water_velocity"
VAR_VO = "northward_sea_water_velocity"

LEGACY_MEASUREMENTS = [VAR_ZOS, VAR_TEMP, VAR_SAL, VAR_UO, VAR_VO]

OBS_TYPE_ARGO = 1
OBS_TYPE_DRIFTER_SST = 2
OBS_TYPE_DRIFTER_CURRENT = 3
OBS_TYPE_SLA = 4

OBS_TYPE_GROUP = {
    OBS_TYPE_ARGO: "ts",
    OBS_TYPE_DRIFTER_SST: "sst",
    OBS_TYPE_DRIFTER_CURRENT: "cur",
    OBS_TYPE_SLA: "sla",
}

QC_MISSING = np.int8(9)
FILL_INT32 = np.int32(-1)

STR_WIDTHS = {
    "obs_id": 96,
    "platform_code": 32,
    # SOURCE carries free text such as "drifting subsurface profiling float".
    "platform_source": 64,
    "sla_mission": 8,
    "qc_reason": 48,
    "data_mode": 1,
}

# Full ordered column set of the intermediate pandas frame.
FLOAT_COLUMNS = [
    VAR_DEPTH,
    VAR_LAT,
    VAR_LON,
    VAR_ZOS,
    VAR_TEMP,
    VAR_SAL,
    VAR_UO,
    VAR_VO,
    "temp_raw",
    "psal_raw",
    "temp_adjusted",
    "psal_adjusted",
    "uo_raw",
    "vo_raw",
    "uo_ws",
    "vo_ws",
    "sla_unfiltered",
]

INT8_COLUMNS = [
    "obs_type",
    "temp_qc",
    "psal_qc",
    "deph_qc",
    "position_qc",
    "time_qc",
    "temp_adjusted_qc",
    "psal_adjusted_qc",
    "uo_qc",
    "vo_qc",
    "ws_type",
    "drogued",
    "sla_flag_keep",
    "qc_keep",
]

INT32_COLUMNS = ["argo_cycle", "current_test"]

STRING_COLUMNS = [
    "obs_id",
    "platform_code",
    "platform_source",
    "data_mode",
    "sla_mission",
    "qc_reason",
]


# ============================================================
# SMALL HELPERS
# ============================================================


def qc_to_int8(values: Any, size: int) -> np.ndarray:
    """Normalise an OceanSITES QC array to int8, 9 for missing.

    Source files use byte flags in the drifter current product and single
    character flags in the temperature and salinity product.
    """
    if values is None:
        return np.full(size, QC_MISSING, dtype=np.int8)
    arr = np.asarray(values)
    if arr.dtype.kind in {"S", "U", "O"}:
        flat = arr.astype(str).ravel()
        out = np.full(flat.size, QC_MISSING, dtype=np.int8)
        for i, token in enumerate(flat):
            token = token.strip()
            if token.isdigit():
                out[i] = np.int8(int(token))
        return out.reshape(arr.shape)
    out = np.where(np.isin(arr, list(range(0, 10))), arr, QC_MISSING)
    return out.astype(np.int8)


def _decode_token(value: Any) -> str:
    """One element of a char or string array as text, no stripping yet.

    xarray hands these back either as numpy bytes or as an object array still
    holding python bytes. str() on bytes yields the repr "b'...'", so the
    decode has to be explicit.
    """
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "ignore")
    return str(value)


def char_to_str(values: Any, size: int) -> np.ndarray:
    """Collapse a (N, STRINGxx) char array or a 1-D array to clean strings."""
    if values is None:
        return np.full(size, "", dtype=object)
    arr = np.asarray(values)
    if arr.dtype.kind == "S":
        arr = np.char.decode(arr, "utf-8", "ignore")
    if arr.ndim == 2:
        joined = np.array(
            ["".join(_decode_token(token) for token in row).strip() for row in arr],
            dtype=object,
        )
        return joined
    return np.array([_decode_token(v).strip() for v in arr.ravel()], dtype=object)


def optional(ds: xr.Dataset, name: str):
    return ds[name].values if name in ds.variables else None


def to_datetime_ns(values: Any) -> np.ndarray:
    """Convert a decoded time array to naive UTC datetime64[ns]."""
    series = pd.to_datetime(pd.Series(np.asarray(values).ravel()), utc=True, errors="coerce")
    return series.dt.tz_localize(None).values.astype("datetime64[ns]")


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for module_name in ["numpy", "pandas", "xarray", "zarr", "s3fs", "copernicusmarine"]:
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[module_name] = "not_installed"
    return versions


def empty_frame() -> pd.DataFrame:
    columns = FLOAT_COLUMNS + INT8_COLUMNS + INT32_COLUMNS + STRING_COLUMNS + ["time_ns"]
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})


# ============================================================
# FILESYSTEMS
# ============================================================


def target_storage_options() -> dict[str, Any]:
    key = os.environ.get("CF_KEY") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("CF_SECRET") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        raise SystemExit(
            "missing credentials: set CF_KEY and CF_SECRET (or AWS_ACCESS_KEY_ID "
            "and AWS_SECRET_ACCESS_KEY) in the environment"
        )
    return {
        "key": key,
        "secret": secret,
        "client_kwargs": {
            "endpoint_url": os.environ.get("CF_ENDPOINT", CLOUDFERRO_ENDPOINT),
            "region_name": CLOUDFERRO_REGION,
        },
        "config_kwargs": {
            "s3": {"addressing_style": "path"},
            "max_pool_connections": 64,
            "retries": {"max_attempts": 10},
        },
    }


def is_s3(target: str) -> bool:
    return target.startswith("s3://")


def get_target_fs(target: str):
    if not is_s3(target):
        import fsspec

        return fsspec.filesystem("file"), {}
    import s3fs

    options = target_storage_options()
    return s3fs.S3FileSystem(**options), options


def get_source_fs():
    import s3fs

    return s3fs.S3FileSystem(
        anon=True,
        client_kwargs={"endpoint_url": SOURCE_ENDPOINT, "region_name": CLOUDFERRO_REGION},
        config_kwargs={"s3": {"addressing_style": "path"}, "max_pool_connections": 32},
    )


def strip_scheme(path: str) -> str:
    return path[len("s3://") :] if path.startswith("s3://") else path


def source_key(product: str, dataset: str, date: dt.date, filename: str) -> str:
    return f"native/{product}/{dataset}/{date:%Y}/{date:%m}/{filename}"


def download_source_file(source_fs, key: str, local_path: Path) -> dict[str, Any] | None:
    """Download one source object and return its provenance record."""
    remote = f"{SOURCE_BUCKET}/{key}"
    started = dt.datetime.now(dt.timezone.utc)
    try:
        info = source_fs.info(remote)
        source_fs.get(remote, str(local_path))
    except Exception as exc:
        logger.warning("download failed for %s: %s", key, exc)
        return None
    return {
        "key": key,
        "size": int(info.get("size", local_path.stat().st_size)),
        "etag": str(info.get("ETag", "")).strip('"'),
        "download_time_utc": started.isoformat(),
        "local_name": local_path.name,
    }


def archive_file(local_path: Path, archive_dir: Path | None, date: dt.date) -> None:
    if archive_dir is None:
        return
    destination = Path(archive_dir) / f"{date:%Y%m%d}"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, destination / local_path.name)


# ============================================================
# EXTRACTORS, one per stream, mirroring the notebook structure
# ============================================================


def _base_frame(n: int, obs_type: int) -> dict[str, Any]:
    """Column skeleton with the fill values every stream starts from."""
    nan = np.full(n, np.nan, dtype=np.float64)
    missing_qc = np.full(n, QC_MISSING, dtype=np.int8)
    return {
        VAR_DEPTH: nan.copy(),
        VAR_LAT: nan.copy(),
        VAR_LON: nan.copy(),
        VAR_ZOS: nan.copy(),
        VAR_TEMP: nan.copy(),
        VAR_SAL: nan.copy(),
        VAR_UO: nan.copy(),
        VAR_VO: nan.copy(),
        "temp_raw": nan.copy(),
        "psal_raw": nan.copy(),
        "temp_adjusted": nan.copy(),
        "psal_adjusted": nan.copy(),
        "uo_raw": nan.copy(),
        "vo_raw": nan.copy(),
        "uo_ws": nan.copy(),
        "vo_ws": nan.copy(),
        "sla_unfiltered": nan.copy(),
        "obs_type": np.full(n, obs_type, dtype=np.int8),
        "temp_qc": missing_qc.copy(),
        "psal_qc": missing_qc.copy(),
        "deph_qc": missing_qc.copy(),
        "position_qc": missing_qc.copy(),
        "time_qc": missing_qc.copy(),
        "temp_adjusted_qc": missing_qc.copy(),
        "psal_adjusted_qc": missing_qc.copy(),
        "uo_qc": missing_qc.copy(),
        "vo_qc": missing_qc.copy(),
        "ws_type": np.full(n, -1, dtype=np.int8),
        "drogued": np.full(n, -1, dtype=np.int8),
        "sla_flag_keep": np.full(n, -1, dtype=np.int8),
        "qc_keep": np.zeros(n, dtype=np.int8),
        "argo_cycle": np.full(n, FILL_INT32, dtype=np.int32),
        "current_test": np.full(n, FILL_INT32, dtype=np.int32),
        "platform_code": np.full(n, "", dtype=object),
        "platform_source": np.full(n, "", dtype=object),
        "data_mode": np.full(n, "", dtype=object),
        "sla_mission": np.full(n, "", dtype=object),
        "qc_reason": np.full(n, "", dtype=object),
        "obs_id": np.full(n, "", dtype=object),
    }


def extract_currents(nc_path: Path) -> pd.DataFrame:
    """Drifter velocities from GL_TS_DC_{YYYYMMDD}_FILTR.nc.

    The DEPTH dimension has length 1 in this product, so column 0 is taken as
    in the notebook. Both the filtered and the unfiltered components are kept.
    """
    ds = xr.open_dataset(nc_path)
    try:
        n = ds.sizes["TIME"]
        columns = _base_frame(n, OBS_TYPE_DRIFTER_CURRENT)

        depth = np.asarray(ds["DEPH"].values)[:, 0].astype(np.float64)
        ewct = np.asarray(ds["EWCT"].values)[:, 0].astype(np.float64)
        nsct = np.asarray(ds["NSCT"].values)[:, 0].astype(np.float64)

        if POLICY["current_basis"] == "filtr" and "EWCT_FILTR" in ds.variables:
            uo = np.asarray(ds["EWCT_FILTR"].values)[:, 0].astype(np.float64)
            vo = np.asarray(ds["NSCT_FILTR"].values)[:, 0].astype(np.float64)
            uo_qc = qc_to_int8(optional(ds, "EWCT_FILTR_QC"), n * 1).reshape(n, -1)[:, 0]
            vo_qc = qc_to_int8(optional(ds, "NSCT_FILTR_QC"), n * 1).reshape(n, -1)[:, 0]
        else:
            uo, vo = ewct, nsct
            uo_qc = qc_to_int8(optional(ds, "EWCT_QC"), n).reshape(n, -1)[:, 0]
            vo_qc = qc_to_int8(optional(ds, "NSCT_QC"), n).reshape(n, -1)[:, 0]

        columns[VAR_DEPTH] = depth
        columns[VAR_LAT] = np.asarray(ds["LATITUDE"].values).astype(np.float64)
        columns[VAR_LON] = np.asarray(ds["LONGITUDE"].values).astype(np.float64)
        columns[VAR_UO] = uo
        columns[VAR_VO] = vo
        columns["uo_raw"] = ewct
        columns["vo_raw"] = nsct
        columns["uo_qc"] = uo_qc
        columns["vo_qc"] = vo_qc
        columns["deph_qc"] = qc_to_int8(optional(ds, "DEPH_QC"), n).reshape(n, -1)[:, 0]
        columns["position_qc"] = qc_to_int8(optional(ds, "POSITION_QC"), n).ravel()
        columns["time_qc"] = qc_to_int8(optional(ds, "TIME_QC"), n).ravel()

        ws_east = optional(ds, "EWCT_WS_FILTR")
        if ws_east is None:
            ws_east = optional(ds, "EWCT_WS")
        ws_north = optional(ds, "NSCT_WS_FILTR")
        if ws_north is None:
            ws_north = optional(ds, "NSCT_WS")
        if ws_east is not None:
            columns["uo_ws"] = np.asarray(ws_east)[:, 0].astype(np.float64)
        if ws_north is not None:
            columns["vo_ws"] = np.asarray(ws_north)[:, 0].astype(np.float64)

        ws_type = optional(ds, "WS_TYPE_OF_PROCESSING")
        if ws_type is not None:
            columns["ws_type"] = np.asarray(ws_type).reshape(n, -1)[:, 0].astype(np.int8)

        current_test = optional(ds, "CURRENT_TEST")
        if current_test is not None:
            raw = np.asarray(current_test).ravel().astype(np.float64)
            filled = np.where(np.isfinite(raw), raw, -1).astype(np.int32)
            columns["current_test"] = filled
            drogued = np.full(n, -1, dtype=np.int8)
            known = filled >= 0
            drogued[known] = 1
            drogued[np.isin(filled, POLICY["drop_undrogued_current_test"])] = 0
            columns["drogued"] = drogued

        columns["platform_code"] = char_to_str(optional(ds, "PLATFORM_CODE"), n)
        columns["platform_source"] = char_to_str(optional(ds, "SOURCE"), n)

        frame = pd.DataFrame(columns)
        frame["time_ns"] = to_datetime_ns(ds["TIME"].values)
        return frame
    finally:
        ds.close()


def _extract_profile_like(nc_path: Path, obs_type: int, with_salinity: bool) -> pd.DataFrame:
    """Shared reader for the (N_PROF, N_LEVELS) temperature and salinity files.

    CO_PR_PF (Argo profiles) and CO_TS_DB (drifter surface temperature) share
    this layout. Every level with a finite depth and at least one finite
    measurement becomes one row, whatever its QC flag.
    """
    ds = xr.open_dataset(nc_path)
    try:
        n_prof = ds.sizes["N_PROF"]
        n_lev = ds.sizes["N_LEVELS"]

        depth = np.asarray(ds["DEPH"].values, dtype=np.float64)
        temp = np.asarray(ds["TEMP"].values, dtype=np.float64)
        temp_qc = qc_to_int8(optional(ds, "TEMP_QC"), n_prof * n_lev).reshape(n_prof, n_lev)
        deph_qc = qc_to_int8(optional(ds, "DEPH_QC"), n_prof * n_lev).reshape(n_prof, n_lev)

        if with_salinity and "PSAL" in ds.variables:
            psal = np.asarray(ds["PSAL"].values, dtype=np.float64)
            psal_qc = qc_to_int8(optional(ds, "PSAL_QC"), n_prof * n_lev).reshape(n_prof, n_lev)
        else:
            psal = np.full_like(temp, np.nan)
            psal_qc = np.full((n_prof, n_lev), QC_MISSING, dtype=np.int8)

        temp_adj = optional(ds, "TEMP_ADJUSTED")
        psal_adj = optional(ds, "PSAL_ADJUSTED")
        temp_adj_qc = qc_to_int8(optional(ds, "TEMP_ADJUSTED_QC"), n_prof * n_lev)
        psal_adj_qc = qc_to_int8(optional(ds, "PSAL_ADJUSTED_QC"), n_prof * n_lev)

        # A row exists if the depth is finite and at least one value is finite.
        selectable = np.isfinite(depth) & (np.isfinite(temp) | np.isfinite(psal))
        pi, li = np.where(selectable)
        n = pi.size
        if n == 0:
            return empty_frame()

        columns = _base_frame(n, obs_type)
        columns[VAR_DEPTH] = depth[pi, li]
        columns[VAR_LAT] = np.asarray(ds["LATITUDE"].values, dtype=np.float64)[pi]
        columns[VAR_LON] = np.asarray(ds["LONGITUDE"].values, dtype=np.float64)[pi]
        columns["temp_raw"] = temp[pi, li]
        columns["psal_raw"] = psal[pi, li]
        columns[VAR_TEMP] = temp[pi, li]
        columns[VAR_SAL] = psal[pi, li]
        columns["temp_qc"] = temp_qc[pi, li]
        columns["psal_qc"] = psal_qc[pi, li]
        columns["deph_qc"] = deph_qc[pi, li]
        columns["position_qc"] = qc_to_int8(optional(ds, "POSITION_QC"), n_prof).ravel()[pi]
        columns["time_qc"] = qc_to_int8(optional(ds, "JULD_QC"), n_prof).ravel()[pi]

        if temp_adj is not None:
            columns["temp_adjusted"] = np.asarray(temp_adj, dtype=np.float64)[pi, li]
            columns["temp_adjusted_qc"] = temp_adj_qc.reshape(n_prof, n_lev)[pi, li]
        if psal_adj is not None:
            columns["psal_adjusted"] = np.asarray(psal_adj, dtype=np.float64)[pi, li]
            columns["psal_adjusted_qc"] = psal_adj_qc.reshape(n_prof, n_lev)[pi, li]

        platform = optional(ds, "PLATFORM_CODE")
        if platform is None:
            platform = optional(ds, "PLATFORM_NUMBER")
        columns["platform_code"] = char_to_str(platform, n_prof)[pi]

        # WMO_INST_TYPE is present but blank in the Coriolis MERC files, so an
        # "is it absent" fallback never fires. Fall back on emptiness instead.
        source_text = char_to_str(optional(ds, "WMO_INST_TYPE"), n_prof)
        if not any(str(v).strip() for v in source_text):
            source_text = char_to_str(optional(ds, "SOURCE"), n_prof)
        columns["platform_source"] = source_text[pi]

        data_mode = optional(ds, "DATA_MODE")
        if data_mode is not None:
            columns["data_mode"] = char_to_str(data_mode, n_prof)[pi]

        cycle = optional(ds, "CYCLE_NUMBER")
        if cycle is not None:
            raw_cycle = np.asarray(cycle).ravel().astype(np.float64)
            columns["argo_cycle"] = np.where(np.isfinite(raw_cycle), raw_cycle, -1).astype(np.int32)[pi]

        frame = pd.DataFrame(columns)
        frame["time_ns"] = to_datetime_ns(ds["JULD"].values)[pi]
        return frame
    finally:
        ds.close()


def extract_profiles(nc_path: Path) -> pd.DataFrame:
    """Argo temperature and salinity profiles from CO_PR_PF_{YYYYMMDD}_MERC.nc."""
    return _extract_profile_like(nc_path, OBS_TYPE_ARGO, with_salinity=True)


def extract_drifter_sst(nc_path: Path) -> pd.DataFrame:
    """Drifter surface temperature from CO_TS_DB_{YYYYMMDD}_MERC.nc."""
    return _extract_profile_like(nc_path, OBS_TYPE_DRIFTER_SST, with_salinity=False)


def extract_sla_nc(nc_path: Path, mission: str) -> pd.DataFrame:
    """One DUACS L3 along-track file. Filtered and unfiltered SLA are both kept."""
    ds = xr.open_dataset(nc_path)
    try:
        if "sla_filtered" in ds.variables:
            filtered_name = "sla_filtered"
        elif "sla" in ds.variables:
            filtered_name = "sla"
        else:
            logger.warning("no SLA variable in %s", nc_path.name)
            return empty_frame()

        sla = np.asarray(ds[filtered_name].values, dtype=np.float64).ravel()
        keep = np.isfinite(sla)
        n = int(keep.sum())
        if n == 0:
            return empty_frame()

        columns = _base_frame(n, OBS_TYPE_SLA)
        columns[VAR_LAT] = np.asarray(ds["latitude"].values, dtype=np.float64).ravel()[keep]
        columns[VAR_LON] = np.asarray(ds["longitude"].values, dtype=np.float64).ravel()[keep]
        columns[VAR_DEPTH] = np.zeros(n, dtype=np.float64)
        columns[VAR_ZOS] = sla[keep]

        unfiltered = optional(ds, "sla_unfiltered")
        if unfiltered is not None:
            columns["sla_unfiltered"] = np.asarray(unfiltered, dtype=np.float64).ravel()[keep]

        columns["sla_mission"] = np.full(n, mission, dtype=object)

        track = optional(ds, "track")
        cycle = optional(ds, "cycle")
        if track is not None and cycle is not None:
            tracks = np.asarray(track).ravel()[keep].astype(np.int64)
            cycles = np.asarray(cycle).ravel()[keep].astype(np.int64)
            columns["platform_code"] = np.array([f"{mission}_c{c}_t{t}" for c, t in zip(cycles, tracks)], dtype=object)
        else:
            columns["platform_code"] = np.full(n, mission, dtype=object)
        columns["platform_source"] = np.full(n, "duacs_l3_my", dtype=object)

        frame = pd.DataFrame(columns)
        frame["time_ns"] = to_datetime_ns(ds["time"].values)[keep]
        return frame
    finally:
        ds.close()


def extract_sla(
    date: dt.date, tmp_dir: Path, archive_dir: Path | None, retries: int = 3
) -> tuple[pd.DataFrame, list[dict], list[str]]:
    """Download and parse every DUACS L3 satellite for one day."""
    import copernicusmarine

    frames: list[pd.DataFrame] = []
    files: list[dict] = []
    satellites_found: list[str] = []

    for mission, dataset_id in SLA_SATELLITES.items():
        for attempt in range(1, retries + 1):
            try:
                before = set(tmp_dir.glob("*.nc"))
                # DUACS names a file dt_global_{mission}_phy_l3_1hz_{measurement
                # date}_{production date}.nc, so a bare *{date}* also matches
                # every file merely produced on that date, which is thousands of
                # files from other measurement days. Anchoring on the separators
                # keeps only the measurement-date field, because the production
                # date is followed by ".nc" rather than "_".
                copernicusmarine.get(
                    dataset_id=dataset_id,
                    filter=f"*_{date:%Y%m%d}_*",
                    output_directory=str(tmp_dir),
                    overwrite=True,
                    no_directories=True,
                )
                new_files = sorted(set(tmp_dir.glob("*.nc")) - before)
                if not new_files:
                    logger.warning("SLA %s: no file for %s", mission, date)
                    break

                satellites_found.append(mission)
                for path in new_files:
                    files.append(
                        {
                            "key": f"{dataset_id}/{path.name}",
                            "size": path.stat().st_size,
                            "etag": sha256_of_file(path)[:32],
                            "etag_kind": "sha256_prefix",
                            "download_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "stream": "sla",
                            "mission": mission,
                        }
                    )
                    frame = extract_sla_nc(path, mission)
                    files[-1]["n_rows"] = int(len(frame))
                    archive_file(path, archive_dir, date)
                    path.unlink(missing_ok=True)
                    if len(frame) > 0:
                        frames.append(frame)
                break
            except Exception as exc:
                logger.warning("SLA %s attempt %d/%d failed: %s", mission, attempt, retries, exc)

    if not frames:
        return empty_frame(), files, satellites_found
    return pd.concat(frames, ignore_index=True), files, satellites_found


# ============================================================
# LONGITUDE CONVENTION
# ============================================================


def normalize_longitude(frame: pd.DataFrame) -> pd.DataFrame:
    """Put longitude on [-180, 180) for every stream.

    The in-situ sources publish longitude on -180..180 but the DUACS L3
    along-track files publish it on 0..360, so SLA rows east of 180 fall off a
    -180..180 model grid and score as missing. Applied once on the combined
    frame so every stream shares one convention. Idempotent.

    The wrap is a single exact addition or subtraction of 360 rather than a
    modulo, so a value already inside the range keeps its exact bits: a modulo
    round-trips through a larger magnitude and perturbs in-range values by an
    ulp. A value outside [-540, 540) would need more than one step and is
    treated as a source that no longer matches the assumed convention.
    """
    if len(frame) == 0:
        return frame
    lon = frame[VAR_LON].to_numpy(dtype=np.float64, copy=True)
    finite = np.isfinite(lon)
    lon[finite & (lon >= 180.0)] -= 360.0
    lon[finite & (lon < -180.0)] += 360.0
    still_out = finite & ((lon >= 180.0) | (lon < -180.0))
    if still_out.any():
        raise GateFailure(
            f"longitude outside [-540, 540) on {int(still_out.sum())} rows, "
            f"worst {float(np.nanmax(np.abs(lon[still_out])))}"
        )
    frame[VAR_LON] = lon
    return frame


# ============================================================
# POLICY, a pure function over the combined frame
# ============================================================


def apply_policy(frame: pd.DataFrame, date: dt.date, policy: dict[str, Any]) -> pd.DataFrame:
    """Set qc_keep, qc_reason and blank the legacy columns of failing rows.

    Pure: it only reads the frame and the policy, and returns a new frame.
    """
    frame = frame.copy()
    n = len(frame)
    if n == 0:
        return frame

    accepted = np.asarray(policy["accepted_qc_flags"], dtype=np.int8)
    # As built, the position check below reads "accepted", not this list, so the
    # store is stricter on position QC than the policy declares. Kept unused so
    # that what the published store was written with stays visible here.
    accepted_pos = np.asarray(policy["accepted_position_qc_flags"], dtype=np.int8)  # noqa: F841
    accepted_time = np.asarray(policy["accepted_time_qc_flags"], dtype=np.int8)
    accepted_depth = np.asarray(policy["accepted_depth_qc_flags"], dtype=np.int8)

    obs_type = frame["obs_type"].to_numpy()
    reason = np.full(n, "", dtype=object)
    keep = np.ones(n, dtype=bool)

    def fail(mask: np.ndarray, label: str) -> None:
        nonlocal keep
        newly = mask & keep
        reason[newly] = label
        keep = keep & ~mask

    # Position and time QC apply to every in-situ row. Satellite rows carry
    # QC 9 for these because the L3 product has no such flags.
    insitu = obs_type != OBS_TYPE_SLA
    fail(insitu & ~np.isin(frame["position_qc"].to_numpy(), accepted), "position_qc")
    fail(insitu & ~np.isin(frame["time_qc"].to_numpy(), accepted_time), "time_qc")

    # Day alignment.
    if policy["enforce_day_alignment"]:
        day_start = np.datetime64(f"{date:%Y-%m-%d}T00:00:00", "ns")
        day_end = day_start + np.timedelta64(1, "D")
        times = frame["time_ns"].to_numpy().astype("datetime64[ns]")
        outside = ~((times >= day_start) & (times < day_end))
        fail(outside | pd.isna(frame["time_ns"]).to_numpy(), "day_misaligned")

    # Argo profiles and drifter SST.
    ts_rows = np.isin(obs_type, [OBS_TYPE_ARGO, OBS_TYPE_DRIFTER_SST])
    fail(ts_rows & ~np.isin(frame["deph_qc"].to_numpy(), accepted_depth), "deph_qc")
    temp_ok = np.isin(frame["temp_qc"].to_numpy(), accepted) & np.isfinite(frame["temp_raw"].to_numpy())
    psal_ok = np.isin(frame["psal_qc"].to_numpy(), accepted) & np.isfinite(frame["psal_raw"].to_numpy())
    fail(ts_rows & ~(temp_ok | psal_ok), "temp_psal_qc")

    # Drifter currents.
    cur_rows = obs_type == OBS_TYPE_DRIFTER_CURRENT
    uo_ok = np.isin(frame["uo_qc"].to_numpy(), accepted)
    vo_ok = np.isin(frame["vo_qc"].to_numpy(), accepted)
    finite_uv = np.isfinite(frame[VAR_UO].to_numpy()) & np.isfinite(frame[VAR_VO].to_numpy())
    fail(cur_rows & ~(uo_ok & vo_ok & finite_uv), "current_qc")
    undrogued = np.isin(frame["current_test"].to_numpy(), policy["drop_undrogued_current_test"])
    fail(cur_rows & undrogued, "undrogued")
    if not policy["keep_unknown_drogue"]:
        fail(cur_rows & (frame["drogued"].to_numpy() < 0), "drogue_unknown")

    # SLA bound.
    sla_rows = obs_type == OBS_TYPE_SLA
    sla_values = frame[VAR_ZOS].to_numpy()
    within = np.abs(sla_values) <= float(policy["sla_abs_max_m"])
    frame.loc[sla_rows, "sla_flag_keep"] = within[sla_rows].astype(np.int8)
    fail(sla_rows & ~(within & np.isfinite(sla_values)), "sla_out_of_bounds")

    frame["qc_keep"] = keep.astype(np.int8)
    frame["qc_reason"] = reason

    # Legacy measurement columns hold policy-passing values only. Per-variable
    # QC additionally blanks the individual variable inside a kept row.
    nan = np.nan
    frame.loc[~keep, LEGACY_MEASUREMENTS] = nan
    frame.loc[keep & ts_rows & ~temp_ok, VAR_TEMP] = nan
    frame.loc[keep & ts_rows & ~psal_ok, VAR_SAL] = nan
    return frame


# ============================================================
# OBS ID AND DEDUP
# ============================================================


def build_obs_ids(frame: pd.DataFrame) -> pd.Series:
    """obs_id = obs_type:platform_code:isotime:depth:group.

    Collisions inside a day are resolved by appending -N to the second and
    later occurrences, so the identifier stays unique per day and stable for a
    given source row. The key never includes a measured value.
    """
    times = pd.to_datetime(frame["time_ns"]).dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    depth = frame[VAR_DEPTH].fillna(-999.0).map(lambda v: f"{v:.2f}")
    group = frame["obs_type"].map(OBS_TYPE_GROUP).fillna("unk")
    base = (
        frame["obs_type"].astype(str)
        + ":"
        + frame["platform_code"].astype(str)
        + ":"
        + times
        + ":"
        + depth
        + ":"
        + group
    )
    occurrence = base.groupby(base).cumcount()
    return base.where(occurrence == 0, base + "-" + occurrence.astype(str))


def dedup(frame: pd.DataFrame, strategy: str) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate obs_id rows, preferring a policy-passing row."""
    if strategy != "first_kept":
        raise ValueError(f"unsupported dedup strategy: {strategy}")
    before = len(frame)
    ordered = frame.sort_values(["obs_id", "qc_keep"], ascending=[True, False], kind="stable")
    deduped = ordered.drop_duplicates(subset="obs_id", keep="first")
    deduped = deduped.sort_index().reset_index(drop=True)
    return deduped, before - len(deduped)


# ============================================================
# DATASET ASSEMBLY AND WRITE
# ============================================================


def to_fixed_string(series: pd.Series, width: int, name: str) -> np.ndarray:
    """Cast to a fixed-width unicode column, refusing to truncate silently.

    A truncated obs_id stops being unique, so an overflow is a build error
    rather than something to absorb.
    """
    text = series.fillna("").astype(str)
    longest = int(text.str.len().max()) if len(text) else 0
    if longest > width:
        worst = text[text.str.len() == longest].iloc[0]
        raise ValueError(
            f"column '{name}' needs width {longest} but STR_WIDTHS gives {width}; " f"longest value: {worst!r}"
        )
    return text.to_numpy(dtype=f"<U{width}")


def combine_to_dataset(frame: pd.DataFrame, date: dt.date, attrs: dict[str, Any]) -> xr.Dataset:
    n = len(frame)
    time_strings = pd.to_datetime(frame["time_ns"]).dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("").to_numpy(dtype="U19")

    data: dict[str, Any] = {}
    for name in FLOAT_COLUMNS:
        data[name] = ("obs", frame[name].to_numpy(dtype=np.float64))
    data[VAR_TIME] = ("obs", time_strings)
    data["time_ns"] = ("obs", frame["time_ns"].to_numpy().astype("datetime64[ns]"))
    for name in INT8_COLUMNS:
        data[name] = ("obs", frame[name].to_numpy(dtype=np.int8))
    for name in INT32_COLUMNS:
        data[name] = ("obs", frame[name].to_numpy(dtype=np.int32))
    for name in STRING_COLUMNS:
        data[name] = ("obs", to_fixed_string(frame[name], STR_WIDTHS.get(name, 32), name))

    ds = xr.Dataset(data, attrs=attrs)
    return ds.chunk({"obs": min(max(n, 1), 250_000)})


def publish_prefix(fs, tmp_path: str, final_path: str) -> None:
    """Move every key of the written tmp prefix onto the final prefix.

    fsspec's recursive mv on this S3 endpoint raises FileNotFoundError while a
    per-key copy of the exact same objects succeeds, so the keys are enumerated
    and copied one by one.
    """
    keys = fs.find(tmp_path)
    if not keys:
        raise RuntimeError(f"nothing written under {tmp_path}")
    protocols = fs.protocol if isinstance(fs.protocol, tuple) else (fs.protocol,)
    needs_mkdir = "file" in protocols  # object stores create the prefix implicitly
    for key in keys:
        destination = f"{final_path}/{key[len(tmp_path):].lstrip('/')}"
        if needs_mkdir:
            fs.makedirs(destination.rsplit("/", 1)[0], exist_ok=True)
        fs.cp_file(key, destination)
    fs.rm(tmp_path, recursive=True)


def write_dataset(ds: xr.Dataset, target_root: str, date: dt.date, overwrite: bool) -> str:
    """Write to a .tmp suffix then rename, so a partial write never lands."""
    fs, storage_options = get_target_fs(target_root)
    final_url = f"{target_root.rstrip('/')}/{date:%Y%m%d}.zarr"
    tmp_url = f"{final_url}.tmp"

    if fs.exists(strip_scheme(final_url)):
        if not overwrite:
            raise RuntimeError(f"target exists and --overwrite not given: {final_url}")
        logger.info("overwrite requested, existing target will be replaced: %s", final_url)

    if fs.exists(strip_scheme(tmp_url)):
        fs.rm(strip_scheme(tmp_url), recursive=True)

    encoding: dict[str, Any] = {VAR_TIME: {"dtype": "U19", "_FillValue": None}}
    for name in STRING_COLUMNS:
        encoding[name] = {"dtype": f"U{STR_WIDTHS.get(name, 32)}", "_FillValue": None}

    kwargs: dict[str, Any] = {
        "mode": "w",
        "consolidated": True,
        "encoding": encoding,
    }
    if storage_options:
        kwargs["storage_options"] = storage_options
    try:
        ds.to_zarr(tmp_url, zarr_format=2, **kwargs)
    except TypeError:
        ds.to_zarr(tmp_url, zarr_version=2, **kwargs)

    if fs.exists(strip_scheme(final_url)):
        fs.rm(strip_scheme(final_url), recursive=True)
    publish_prefix(fs, strip_scheme(tmp_url), strip_scheme(final_url))
    return final_url


def write_manifest(manifest: dict[str, Any], target_root: str, date: dt.date) -> str:
    fs, _ = get_target_fs(target_root)
    url = f"{target_root.rstrip('/')}/{date:%Y%m%d}.manifest.json"
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode()
    path = strip_scheme(url)
    parent = os.path.dirname(path)
    if not is_s3(target_root):
        Path(parent).mkdir(parents=True, exist_ok=True)
    with fs.open(path, "wb") as handle:
        handle.write(payload)
    return url


def manifest_exists_and_valid(target_root: str, date: dt.date) -> bool:
    fs, _ = get_target_fs(target_root)
    manifest_path = strip_scheme(f"{target_root.rstrip('/')}/{date:%Y%m%d}.manifest.json")
    zarr_path = strip_scheme(f"{target_root.rstrip('/')}/{date:%Y%m%d}.zarr")
    if not fs.exists(manifest_path):
        return False
    try:
        with fs.open(manifest_path, "rb") as handle:
            manifest = json.loads(handle.read().decode())
    except Exception:
        return False
    if manifest.get("status") != "written":
        return False
    return any(fs.exists(f"{zarr_path}/{marker}") for marker in [".zmetadata", ".zgroup"])


# ============================================================
# GATES
# ============================================================


class GateFailure(RuntimeError):
    pass


def check_gates(
    date: dt.date,
    stream_files: dict[str, list[dict]],
    satellites_found: list[str],
    min_satellites: int,
    allow_missing_sla: bool,
) -> None:
    for stream in ["currents", "profiles", "drifter_sst"]:
        if not stream_files.get(stream):
            raise GateFailure(f"{date:%Y%m%d}: in-situ stream '{stream}' has zero files")

    age_days = (dt.date.today() - date).days
    if len(satellites_found) < min_satellites:
        if age_days > RECENT_DAYS:
            raise GateFailure(
                f"{date:%Y%m%d}: only {len(satellites_found)} SLA satellites found "
                f"({min_satellites} required for a date older than {RECENT_DAYS} days)"
            )
        if not allow_missing_sla:
            raise GateFailure(
                f"{date:%Y%m%d}: only {len(satellites_found)} SLA satellites found for a "
                "recent date; pass --allow-missing-sla to accept this"
            )
        logger.warning(
            "%s: accepting %d SLA satellites because --allow-missing-sla is set",
            f"{date:%Y%m%d}",
            len(satellites_found),
        )


# ============================================================
# PER-DAY DRIVER
# ============================================================


def process_day(date: dt.date, args: argparse.Namespace, script_sha: str) -> dict[str, Any]:
    date_str = f"{date:%Y%m%d}"

    if not args.overwrite and manifest_exists_and_valid(args.target, date):
        logger.info("%s: manifest present and valid, skipping", date_str)
        return {"date": date_str, "status": "skipped_exists"}

    source_fs = get_source_fs()
    archive_dir = Path(args.archive_dir) if args.archive_dir else None
    stream_files: dict[str, list[dict]] = {}
    counts_before: dict[str, int] = {}

    with tempfile.TemporaryDirectory(dir=args.tmp_root) as tmp:
        tmp_dir = Path(tmp)
        frames: list[pd.DataFrame] = []

        specs = [
            ("currents", f"GL_TS_DC_{date_str}_FILTR.nc", CURRENTS_PRODUCT, CURRENTS_DATASET, extract_currents),
            ("profiles", f"CO_PR_PF_{date_str}_MERC.nc", TS_PRODUCT, TS_DATASET, extract_profiles),
            ("drifter_sst", f"CO_TS_DB_{date_str}_MERC.nc", TS_PRODUCT, TS_DATASET, extract_drifter_sst),
        ]

        for stream, filename, product, dataset, extractor in specs:
            local = tmp_dir / filename
            record = download_source_file(source_fs, source_key(product, dataset, date, filename), local)
            if record is None:
                stream_files[stream] = []
                counts_before[stream] = 0
                continue
            frame = extractor(local)
            record["stream"] = stream
            record["n_rows"] = int(len(frame))
            stream_files[stream] = [record]
            counts_before[stream] = int(len(frame))
            archive_file(local, archive_dir, date)
            local.unlink(missing_ok=True)
            logger.info("%s: %s -> %d rows", date_str, stream, len(frame))
            if len(frame) > 0:
                frames.append(frame)

        sla_frame, sla_files, satellites_found = extract_sla(date, tmp_dir, archive_dir)
        stream_files["sla"] = sla_files
        counts_before["sla"] = int(len(sla_frame))
        if len(sla_frame) > 0:
            frames.append(sla_frame)
        logger.info("%s: sla -> %d rows from %s", date_str, len(sla_frame), satellites_found)

        check_gates(date, stream_files, satellites_found, args.min_satellites, args.allow_missing_sla)

        if not frames:
            raise GateFailure(f"{date_str}: no rows extracted from any stream")

        combined = pd.concat(frames, ignore_index=True)
        combined = normalize_longitude(combined)
        combined = apply_policy(combined, date, POLICY)
        combined["obs_id"] = build_obs_ids(combined)
        combined, n_duplicates = dedup(combined, POLICY["dedup_strategy"])

        counts_after = {
            name: int(((combined["obs_type"] == code) & (combined["qc_keep"] == 1)).sum())
            for name, code in [
                ("currents", OBS_TYPE_DRIFTER_CURRENT),
                ("profiles", OBS_TYPE_ARGO),
                ("drifter_sst", OBS_TYPE_DRIFTER_SST),
                ("sla", OBS_TYPE_SLA),
            ]
        }

        attrs = {
            "title": f"OceanBench class-4 observations {date:%Y-%m-%d}",
            "date": f"{date:%Y-%m-%d}",
            "obs_basis_version": args.obs_basis_version,
            "build_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "builder_script": Path(__file__).name,
            "builder_script_sha256": script_sha,
            "source_datasets": json.dumps(
                {
                    "currents": {"product": CURRENTS_PRODUCT, "dataset": CURRENTS_DATASET},
                    "temperature_salinity": {"product": TS_PRODUCT, "dataset": TS_DATASET},
                    "sla": SLA_SATELLITES,
                }
            ),
            "sla_satellites_found": json.dumps(satellites_found),
            "source_files": json.dumps(stream_files),
            "package_versions": json.dumps(package_versions()),
            "policy": json.dumps(POLICY),
            "row_counts_before_policy": json.dumps(counts_before),
            "row_counts_after_policy": json.dumps(counts_after),
            "n_duplicates_removed": int(n_duplicates),
            "obs_type_enum": json.dumps({"argo_profile": 1, "drifter_sst": 2, "drifter_current": 3, "sla": 4}),
        }

        ds = combine_to_dataset(combined, date, attrs)
        try:
            url = write_dataset(ds, args.target, date, args.overwrite)
            n_obs = int(ds.sizes["obs"])
        finally:
            ds.close()
            del ds
            gc.collect()

        manifest = {
            "date": f"{date:%Y-%m-%d}",
            "status": "written",
            "target": url,
            "obs_basis_version": args.obs_basis_version,
            "build_timestamp_utc": attrs["build_timestamp_utc"],
            "builder_script_sha256": script_sha,
            "policy": POLICY,
            "source_files": stream_files,
            "sla_satellites_found": satellites_found,
            "row_counts_before_policy": counts_before,
            "row_counts_after_policy": counts_after,
            "n_duplicates_removed": int(n_duplicates),
            "n_obs_total": n_obs,
            "n_obs_kept": int((combined["qc_keep"] == 1).sum()),
            "package_versions": package_versions(),
        }
        manifest_url = write_manifest(manifest, args.target, date)

        logger.info("%s: written %d rows to %s", date_str, n_obs, url)
        return {
            "date": date_str,
            "status": "written",
            "n_obs": n_obs,
            "n_kept": manifest["n_obs_kept"],
            "target": url,
            "manifest": manifest_url,
        }


def _worker(payload: tuple[str, dict, str]) -> dict[str, Any]:
    date_iso, args_dict, script_sha = payload
    args = argparse.Namespace(**args_dict)
    date = dt.date.fromisoformat(date_iso)
    try:
        return process_day(date, args, script_sha)
    except Exception as exc:
        logger.error("%s: failed: %s: %s", date_iso, type(exc).__name__, exc)
        return {"date": date_iso, "status": f"failed: {type(exc).__name__}: {exc}"}


# ============================================================
# CLI
# ============================================================


def parse_dates(args: argparse.Namespace) -> list[dt.date]:
    if args.dates:
        return [pd.Timestamp(d).date() for d in args.dates]
    if not (args.start and args.end):
        raise SystemExit("give either --dates or both --start and --end")
    return [ts.date() for ts in pd.date_range(args.start, args.end, freq="D")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", help="first date, YYYY-MM-DD")
    parser.add_argument("--end", help="last date inclusive, YYYY-MM-DD")
    parser.add_argument("--dates", nargs="+", help="explicit list of dates instead of a range")
    parser.add_argument("--target", default=DEFAULT_TARGET, help=f"output prefix, default {DEFAULT_TARGET}")
    parser.add_argument("--obs-basis-version", default=DEFAULT_OBS_BASIS_VERSION, dest="obs_basis_version")
    parser.add_argument("--archive-dir", default=None, dest="archive_dir", help="keep downloaded source .nc files here")
    parser.add_argument("--tmp-root", default=None, dest="tmp_root", help="parent directory for scratch space")
    parser.add_argument("--overwrite", action="store_true", help="rebuild days that already have a valid manifest")
    parser.add_argument("--allow-missing-sla", action="store_true", dest="allow_missing_sla")
    parser.add_argument("--min-satellites", type=int, default=DEFAULT_MIN_SATELLITES, dest="min_satellites")
    parser.add_argument("--workers", type=int, default=1, help="number of day-level worker processes")
    parser.add_argument("--results-csv", default=None, dest="results_csv")
    parser.add_argument("--log-level", default="INFO", dest="log_level")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")

    dates = parse_dates(args)
    script_sha = sha256_of_file(Path(__file__))

    logger.info("dates: %d, first %s, last %s", len(dates), dates[0], dates[-1])
    logger.info("target: %s", args.target)
    logger.info("script sha256: %s", script_sha)

    if args.workers > 1:
        payloads = [(d.isoformat(), vars(args), script_sha) for d in dates]
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_worker, payloads))
    else:
        results = [_worker((d.isoformat(), vars(args), script_sha)) for d in dates]

    if args.results_csv:
        pd.DataFrame(results).to_csv(args.results_csv, index=False)

    failed = [r for r in results if str(r.get("status", "")).startswith("failed")]
    for record in failed:
        logger.error("FAILED %s: %s", record["date"], record["status"])
    logger.info(
        "done: %d written, %d skipped, %d failed",
        sum(1 for r in results if r.get("status") == "written"),
        sum(1 for r in results if r.get("status") == "skipped_exists"),
        len(failed),
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
