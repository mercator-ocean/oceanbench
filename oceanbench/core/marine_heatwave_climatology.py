# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import xarray

from oceanbench.core.resolution import get_dataset_resolution

_CLIMATOLOGY_BASE_URL = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/public/mhw/climatology/"

_CLIMATOLOGY_URL_BY_RESOLUTION = {
    "quarter_degree": _CLIMATOLOGY_BASE_URL + "GLORYS12V1-1993-2022_daily_gridGLONET.zarr",
    "twelfth_degree": _CLIMATOLOGY_BASE_URL + "GLORYS12V1-1993-2022_alldays_gridT_level00_pgs.zarr",
}


def marine_heatwave_climatology_is_available(
    challenger_dataset: xarray.Dataset,
) -> bool:
    return get_dataset_resolution(challenger_dataset) in _CLIMATOLOGY_URL_BY_RESOLUTION


def marine_heatwave_climatology_mean_and_percentile_90(
    challenger_dataset: xarray.Dataset,
) -> tuple[xarray.DataArray, xarray.DataArray]:
    resolution = get_dataset_resolution(challenger_dataset)
    if resolution not in _CLIMATOLOGY_URL_BY_RESOLUTION:
        raise ValueError(
            f"No MHW climatology available for resolution '{resolution}'. "
            f"Available resolutions: {list(_CLIMATOLOGY_URL_BY_RESOLUTION.keys())}"
        )
    dataset = xarray.open_zarr(
        _CLIMATOLOGY_URL_BY_RESOLUTION[resolution],
        consolidated=True,
    )
    climatology_mean = dataset["thetao_mean"]
    percentile_90 = dataset["thetao_pct"]
    if "quantile" in percentile_90.dims:
        percentile_90 = percentile_90.sel(quantile=0.9, method="nearest")
    return climatology_mean, percentile_90
