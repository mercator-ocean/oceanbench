# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Wind slippage magnitude distribution per CURRENT_TEST code, on the scored row set.

Reads the STRAT view, where a finite current means the row is in the scored set
(default policy kept rows plus the restored 011 stratum). Percentiles are exact,
computed over the pooled full year rather than from partial sums.
"""
import os
import numpy
import xarray
from concurrent.futures import ProcessPoolExecutor

VIEW = "/scratch/jseillade/obs-rebuild/views2/STRAT"
UO = "eastward_sea_water_velocity"
VO = "northward_sea_water_velocity"


def day(name):
    dataset = xarray.open_dataset(os.path.join(VIEW, name + ".zarr"), engine="zarr",
                                  decode_cf=False, consolidated=True)
    uo = dataset[UO].values
    vo = dataset[VO].values
    code = dataset["current_test"].values
    ws = dataset["ws_magnitude"].values
    latitude = dataset["latitude"].values
    dataset.close()
    active = numpy.isfinite(uo) & numpy.isfinite(vo)
    return (code[active].astype(numpy.int32), ws[active].astype(numpy.float32),
            numpy.sqrt(uo[active] ** 2 + vo[active] ** 2).astype(numpy.float32),
            latitude[active].astype(numpy.float32))


days = sorted(n[:-5] for n in os.listdir(VIEW) if n.endswith(".zarr"))
codes, slips, speeds, lats = [], [], [], []
with ProcessPoolExecutor(max_workers=12) as pool:
    for c, w, s, la in pool.map(day, days):
        codes.append(c); slips.append(w); speeds.append(s); lats.append(la)
code = numpy.concatenate(codes); ws = numpy.concatenate(slips)
speed = numpy.concatenate(speeds); latitude = numpy.concatenate(lats)
print("days", len(days), "active rows", code.size)
print("code n ws_finite_pct median p90 p99 rms mean_speed abs_lat_mean")
for value in [313, 312, 311, 213, 212, 211, 11]:
    mask = code == value
    finite = mask & numpy.isfinite(ws)
    magnitude = ws[finite]
    print("%d %d %.2f %.5f %.5f %.5f %.5f %.5f %.2f" % (
        value, int(mask.sum()), 100.0 * finite.sum() / max(int(mask.sum()), 1),
        numpy.median(magnitude), numpy.percentile(magnitude, 90),
        numpy.percentile(magnitude, 99), numpy.sqrt((magnitude ** 2).mean()),
        speed[mask].mean(), numpy.abs(latitude[mask]).mean()))
