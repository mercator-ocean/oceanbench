# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# Open GLOW forecasts with xarray.
# GLOW is evaluated locally and is not an official submission: the forecasts live on
# the machine that produced them, so this file opens them from disk rather than from
# a published bucket. All 52 Wednesday challenger starts of 2024, 20240103 through
# 20241225, each initialised from the as-issued GLO12 nowcast of the Tuesday before
# it, quarter degree, 9 lead days.
#
# Nine, not ten: the IFS forecast package that forces the run carries
# lead_day_index 0..9, so a tenth forecast day would ride on persisted lead 9
# forcing rather than on a real forecast. The store on disk still holds ten days per
# start and nothing was recomputed; this module drops the last time step on the way
# in.
#
# Run glowcascade_v4_nofilter: the glowcascade_v4 recipe with the equatorial notch
# turned off, which is the configuration GLOW actually serves. Same two checkpoints,
# lead 1 from the v4 anneal checkpoint at step 87616 and leads 2 to 9 from the v4
# ladder h5 checkpoint at step 1869. Scores sit within 0.012 percent of the notched
# run on the mean of the nine metric tables.
import datetime
import pathlib

import xarray

_ROOT = pathlib.Path("/mnt/data/glonet2/ifs21/forecasts/glowcascade_v4_nofilter")
_PATHS = sorted(_ROOT.glob("2024*.zarr"))
_FIRST_DAYS = [datetime.datetime.strptime(path.stem, "%Y%m%d") for path in _PATHS]


_LEAD_DAYS = 9


def _prepared(dataset: xarray.Dataset) -> xarray.Dataset:
    dataset = dataset.isel(time=slice(0, _LEAD_DAYS))
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
