#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Region-write real 15 m currents (depth idx 9,10 = 13.467 & 15.810 m) into the
existing weekly climatology zarrs, reusing the FIXED assembly logic. Only uo/vo
at those two depths change; every other chunk (thetao/so/zos, other depths) is
physically untouched. Requires assemble_climatology_forecasts.py to already read
8 scored tags (range(8)) so uo/vo are finite at idx9,10 after reindex."""

import argparse

import numpy as np

import assemble_climatology_forecasts as A

DEPTH_IDX = [9, 10]  # GLORYS grid: 13.46714 m, 15.81007 m -> straddle the 15.0 m obs target
EXPECTED_DEPTHS = [13.46714, 15.81007]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-first", action="store_true", help="patch just week 1 (test)")
    args = ap.parse_args()

    master = A.load_master()  # range(8): uo/vo finite at idx0,9,10,17,21,26,28,31
    dates = A.start_dates()
    if args.only_first:
        dates = dates[:1]

    for i, start in enumerate(dates):
        path = "{}/{}.zarr".format(A.OUTPUT, start.strftime("%Y%m%d"))
        week = A.assemble_week(master, start)  # full lazy ds, 50-depth grid
        sub = week[["uo", "vo"]].isel(depth=DEPTH_IDX)  # (time, depth=2, lat, lon)
        dvals = np.asarray(sub["depth"].values, dtype=float)
        assert np.allclose(dvals, EXPECTED_DEPTHS, atol=0.05), "unexpected depths {}".format(dvals)
        sub = sub.drop_vars(list(sub.coords))  # positional region write, data only
        sub = sub.chunk({"time": 1, "depth": 1, "latitude": 640, "longitude": 1280})
        sub.to_zarr(
            path,
            mode="r+",
            region={
                "time": slice(0, 10),
                "depth": slice(9, 11),
                "latitude": slice(0, 2041),
                "longitude": slice(0, 4320),
            },
        )
        print("[{}/{}] {:%Y-%m-%d} patched depth idx {}".format(i + 1, len(dates), start, DEPTH_IDX), flush=True)
    print("PATCH DONE", flush=True)


if __name__ == "__main__":
    main()
