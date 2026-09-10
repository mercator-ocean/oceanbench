# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# Open HClimRep (WeatherGenerator ft0818) forecasts with xarray.
# HClimRep is evaluated locally and is not an official submission: the forecast store
# was downloaded from Hugging Face to the machine that scored it, so this file opens it
# from disk rather than from a published bucket. 52 weekly Wednesday starts of 2024
# (2024-01-03 to 2024-12-25), GLORYS nowcast initial conditions of the preceding
# Tuesday, quarter degree, 10 lead days.
# Source: https://huggingface.co/datasets/kacpnowak/hclimrep-ocean-oceanbench
# Store: ft0818/wg_ft0818_B1.zarr, rechunked to (1, 1, 1, 672, 1440) without any
# change to the values.
import xarray

CHALLENGER_ZARR = "/mnt/shared/jseillade/hclimrep/store/wg_ft0818_B1.zarr"

challenger_dataset: xarray.Dataset = xarray.open_zarr(CHALLENGER_ZARR, consolidated=True)

challenger_dataset
