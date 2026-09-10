# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# Open GLOW forecasts with xarray.
# GLOW is evaluated locally and is not an official submission: the forecasts live on
# the machine that produced them, so this file opens them from disk rather than from
# a published bucket. 51 Tuesday starts of 2024, as-issued GLO12 nowcast initial
# conditions, quarter degree, 10 lead days.
import datetime
import pathlib

import xarray

_ROOT = pathlib.Path("/mnt/data/glonet2/ifs21/forecasts/glowcascade_final")
_PATHS = sorted(_ROOT.glob("2024*.zarr"))
_FIRST_DAYS = [datetime.datetime.strptime(path.stem, "%Y%m%d") for path in _PATHS]


def _prepared(dataset: xarray.Dataset) -> xarray.Dataset:
    lead_count = dataset.sizes["time"]
    return dataset.rename({"time": "lead_day_index"}).assign_coords({"lead_day_index": range(lead_count)})


challenger_dataset: xarray.Dataset = xarray.open_mfdataset(
    [str(path) for path in _PATHS],
    engine="zarr",
    preprocess=_prepared,
    combine="nested",
    concat_dim="first_day_datetime",
    parallel=False,
).assign_coords({"first_day_datetime": _FIRST_DAYS})

challenger_dataset
