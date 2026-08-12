# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Census of current_test codes and wind-slippage coverage over all 370 days."""
import collections, json, os
import numpy, xarray
from concurrent.futures import ProcessPoolExecutor

SRC = "/scratch/jseillade/obs-rebuild/store-v2"
UO = "eastward_sea_water_velocity"
VO = "northward_sea_water_velocity"


def one(day):
    ds = xarray.open_dataset(os.path.join(SRC, f"{day}.zarr"), engine="zarr",
                             decode_cf=False, consolidated=True)
    ot = ds["obs_type"].values
    ct = ds["current_test"].values
    uo = ds[UO].values
    vo = ds[VO].values
    uws = ds["uo_ws"].values
    vws = ds["vo_ws"].values
    ds.close()
    cur = ot == 3
    fin = cur & numpy.isfinite(uo) & numpy.isfinite(vo)
    wsf = numpy.isfinite(uws) & numpy.isfinite(vws)
    return {
        "day": day,
        "rows": int(ot.size),
        "currents": int(cur.sum()),
        "finite": int(fin.sum()),
        "finite_ws": int((fin & wsf).sum()),
        "codes_all": {str(k): int(v) for k, v in collections.Counter(ct[cur].tolist()).items()},
        "codes_finite": {str(k): int(v) for k, v in collections.Counter(ct[fin].tolist()).items()},
        "ws_on_noncurrent": int((~cur & (numpy.isfinite(uws) | numpy.isfinite(vws))).sum()),
        "u_only_finite": int((cur & numpy.isfinite(uo) & ~numpy.isfinite(vo)).sum()),
        "ws_u_only": int((cur & numpy.isfinite(uws) & ~numpy.isfinite(vws)).sum()),
    }


def main():
    days = sorted(n[:-5] for n in os.listdir(SRC) if n.endswith(".zarr"))
    out = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for rec in pool.map(one, days):
            out.append(rec)
    allc, finc = collections.Counter(), collections.Counter()
    tot = collections.Counter()
    for r in out:
        for k, v in r["codes_all"].items():
            allc[k] += v
        for k, v in r["codes_finite"].items():
            finc[k] += v
        for k in ("rows", "currents", "finite", "finite_ws", "ws_on_noncurrent", "u_only_finite", "ws_u_only"):
            tot[k] += r[k]
    print("days", len(out))
    print("totals", dict(tot))
    print("codes_all", dict(sorted(allc.items(), key=lambda x: int(x[0]))))
    print("codes_finite", dict(sorted(finc.items(), key=lambda x: int(x[0]))))
    print("finite_fraction %.5f" % (tot["finite"] / tot["currents"]))
    print("ws_fraction_of_finite %.5f" % (tot["finite_ws"] / tot["finite"]))
    with open("/scratch/jseillade/obs-rebuild/rescore/full-rescore-v21/census_codes.json", "w") as fh:
        json.dump({"per_day": out, "codes_all": dict(allc), "codes_finite": dict(finc),
                   "totals": dict(tot)}, fh, indent=1)


if __name__ == "__main__":
    main()
