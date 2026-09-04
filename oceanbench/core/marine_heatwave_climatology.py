# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path

import xarray

from oceanbench.core.local_stage import (
    local_stage_directory,
    open_or_create_local_stage_dataset,
    should_stage_locally,
    write_dataset_to_local_stage,
)
from oceanbench.core.resolution import get_dataset_resolution

LOCAL_STAGE_REFERENCES_KEY = "references"

_CLIMATOLOGY_BASE_URL = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/public/mhw/climatology/"

_CLIMATOLOGY_URL_BY_RESOLUTION = {
    "quarter_degree": _CLIMATOLOGY_BASE_URL + "GLORYS12V1-1993-2022_daily_gridGLONET.zarr",
    "twelfth_degree": _CLIMATOLOGY_BASE_URL + "GLORYS12V1-1993-2022_alldays_gridT_level00_pgs.zarr",
}


def marine_heatwave_climatology_is_available(
    challenger_dataset: xarray.Dataset,
) -> bool:
    return get_dataset_resolution(challenger_dataset) in _CLIMATOLOGY_URL_BY_RESOLUTION


def _climatology_stage_path(resolution: str) -> Path:
    return local_stage_directory() / f"marine-heatwave-climatology-glorys12-1993-2022-{resolution}.zarr"


def _open_staged_climatology_dataset(stage_path: Path) -> xarray.Dataset:
    return xarray.open_zarr(stage_path)


def _build_staged_climatology_dataset(climatology_url: str, stage_path: Path) -> None:
    climatology_dataset = xarray.open_zarr(climatology_url, consolidated=True)
    try:
        write_dataset_to_local_stage(climatology_dataset, stage_path)
    finally:
        climatology_dataset.close()


def _open_climatology_dataset(resolution: str) -> xarray.Dataset:
    climatology_url = _CLIMATOLOGY_URL_BY_RESOLUTION[resolution]
    if not should_stage_locally(LOCAL_STAGE_REFERENCES_KEY):
        return xarray.open_zarr(climatology_url, consolidated=True)
    return open_or_create_local_stage_dataset(
        _climatology_stage_path(resolution),
        open_staged_dataset=_open_staged_climatology_dataset,
        build_stage=lambda stage_path: _build_staged_climatology_dataset(climatology_url, stage_path),
    )


def marine_heatwave_climatology_mean_and_percentile_90(
    challenger_dataset: xarray.Dataset,
) -> tuple[xarray.DataArray, xarray.DataArray]:
    resolution = get_dataset_resolution(challenger_dataset)
    if resolution not in _CLIMATOLOGY_URL_BY_RESOLUTION:
        raise ValueError(
            f"No MHW climatology available for resolution '{resolution}'. "
            f"Available resolutions: {list(_CLIMATOLOGY_URL_BY_RESOLUTION.keys())}"
        )
    dataset = _open_climatology_dataset(resolution)
    climatology_mean = dataset["thetao_mean"]
    percentile_90 = dataset["thetao_pct"]
    if "quantile" in percentile_90.dims:
        percentile_90 = percentile_90.sel(quantile=0.9, method="nearest")
    return climatology_mean, percentile_90
