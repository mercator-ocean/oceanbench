# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Append the per-code wind slippage table and the cut arithmetic to RESULTS.md."""
import glob, os
import numpy, pandas, xarray
from concurrent.futures import ProcessPoolExecutor

VIEW = "/scratch/jseillade/obs-rebuild/views2/STRAT"
STORE = "/scratch/jseillade/obs-rebuild/store-v2"
OUT = "/scratch/jseillade/obs-rebuild/rescore/code-strata"
UO = "eastward_sea_water_velocity"
VO = "northward_sea_water_velocity"
CODES = [313, 312, 311, 213, 212, 211, 11]
LABEL = {11: "011"}


def day(name):
    view = xarray.open_dataset(os.path.join(VIEW, name + ".zarr"), engine="zarr", decode_cf=False, consolidated=True)
    active = numpy.isfinite(view[UO].values) & numpy.isfinite(view[VO].values)
    code = view["current_test"].values[active].astype(numpy.int32)
    ws = view["ws_magnitude"].values[active].astype(numpy.float32)
    view.close()
    store = xarray.open_dataset(os.path.join(STORE, name + ".zarr"), engine="zarr", decode_cf=False, consolidated=True)
    ws_type = store["ws_type"].values[active].astype(numpy.int8)
    store.close()
    return code, ws, ws_type


days = sorted(n[:-5] for n in os.listdir(VIEW) if n.endswith(".zarr"))
codes, slips, types = [], [], []
with ProcessPoolExecutor(max_workers=12) as pool:
    for c, w, t in pool.map(day, days):
        codes.append(c); slips.append(w); types.append(t)
code = numpy.concatenate(codes); ws = numpy.concatenate(slips); ws_type = numpy.concatenate(types)

lines = ["## Wind slippage magnitude per code, whole year, scored row set", "",
         "Magnitude is sqrt(uo_ws^2 + vo_ws^2) in m/s over the store rows that enter the",
         "scoring set. ws_type is WS_TYPE_OF_PROCESSING: 0 nominal, 1 from_mean,",
         "2 from_climatology, 3 adaptative, -1 absent.", "",
         "| code | store rows scored | finite WS % | median | p90 | p99 | rms | ws_type mix |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
for value in CODES:
    mask = code == value
    finite = mask & numpy.isfinite(ws)
    magnitude = ws[finite]
    unique, counts = numpy.unique(ws_type[mask], return_counts=True)
    mix = " ".join(f"{int(u)}:{100.0*c/mask.sum():.0f}%" for u, c in zip(unique, counts))
    lines.append("| " + " | ".join([
        LABEL.get(value, str(value)), str(int(mask.sum())),
        f"{100.0*finite.sum()/mask.sum():.2f}",
        f"{numpy.median(magnitude):.5f}", f"{numpy.percentile(magnitude, 90):.5f}",
        f"{numpy.percentile(magnitude, 99):.5f}", f"{numpy.sqrt((magnitude**2).mean()):.5f}", mix]) + " |")
lines.append("")

frame = pandas.concat([pandas.read_csv(f) for f in sorted(glob.glob(
    "/scratch/jseillade/obs-rebuild/rescore/results/glonet-strata/*.csv"))], ignore_index=True)
frame["lead"] = frame["lead_day"].astype(int) + 1
frame = frame[(frame["lead"] >= 1) & (frame["lead"] <= 9) & (frame["variable"] == "uo")]
matchups = frame.groupby("current_test")["count"].sum()
kept_rows = {int(v): int((code == v).sum()) for v in CODES}
kept_total = sum(n for v, n in kept_rows.items() if v != 11)
matchup_total = int(matchups[matchups.index != 11].sum())

lines += ["## What a confidence cut would remove", "",
          "Percentages are of the default policy kept set (011 already excluded).", "",
          "| cut | codes dropped | store rows removed | % of kept store rows | matchups removed | % of matchups |",
          "| --- | --- | --- | --- | --- | --- |"]
cuts = [
    ("strong only", [312, 311, 213, 212, 211]),
    ("drop weak wind correlation", [312, 212]),
    ("drop wind test not performed", [311, 211]),
    ("drop weak submersion", [213, 212, 211]),
    ("drop 211 only", [211]),
    ("drop 211 and 213", [211, 213]),
]
for name, dropped in cuts:
    rows = sum(kept_rows[c] for c in dropped)
    match = int(matchups[matchups.index.isin(dropped)].sum())
    lines.append("| " + " | ".join([
        name, " ".join(str(c) for c in dropped), str(rows),
        f"{100.0*rows/kept_total:.2f}%", str(match), f"{100.0*match/matchup_total:.2f}%"]) + " |")
lines.append("")

with open(os.path.join(OUT, "RESULTS.md"), "a") as handle:
    handle.write("\n".join(lines) + "\n")
print("\n".join(lines))
