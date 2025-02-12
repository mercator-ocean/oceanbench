import datetime
import multiprocessing
from pathlib import Path
from typing import Any, List

import numpy
import xarray


def _get_wednesdays(year: int) -> List[datetime.date]:
    d = datetime.date(year, 1, 1)
    d += datetime.timedelta(days=(2 - d.weekday() + 7) % 7)
    wednesdays = []
    while d.year == year:
        wednesdays.append(d)
        d += datetime.timedelta(weeks=1)
    return wednesdays


def get_rmse_glonet(forecast, ref, var, lead, level):
    cpu_count = multiprocessing.cpu_count()
    with multiprocessing.Pool(cpu_count) as _:
        if var == "zos":
            mask = ~numpy.isnan(forecast[var][lead]) & ~numpy.isnan(ref[var][level, lead])
            rmse = numpy.sqrt(numpy.mean((forecast[var][lead].data[mask] - ref[var][level, lead].data[mask]) ** 2))
        else:
            mask = ~numpy.isnan(forecast[var][lead, level].data) & ~numpy.isnan(ref[var][lead, level].data)
            rmse = numpy.sqrt(
                numpy.mean((forecast[var][lead, level].data[mask] - ref[var][lead, level].data[mask]) ** 2)
            )
    return rmse


def get_glonet_rmse_for_given_days(
    depthg,
    var,
    days: List[datetime.date],
    glonet_datasets_path: Path,
    glorys_datasets_path: Path,
):
    j = 0
    nweeks = 1
    aa = numpy.zeros((nweeks, 10))

    for day in days:
        glonet = xarray.open_dataset(glonet_datasets_path / f"{str(day)}.nc", engine="netcdf4")
        glonetr = xarray.open_dataset(glorys_datasets_path / f"{str(day)}.nc", engine="netcdf4")
        for i in range(0, 10):
            aa[j, i] = get_rmse_glonet(glonet, glonetr, var, i, depthg)
        j = j + 1
        if j > nweeks - 1:
            break
    glonet_rmse = aa.mean(axis=0)
    return glonet_rmse


def glonet_pointwise_evaluation_core(glonet_datasets_path: Path, glorys_datasets_path: Path) -> numpy.ndarray[Any]:
    wednesdays_2024 = _get_wednesdays(2024)

    gnet = {"uo": [], "vo": [], "so": [], "thetao": [], "zos": []}
    variables_withouth_zos = ["uo", "vo", "so", "thetao"]
    mindepth = 0
    maxdepth = 21
    for depth in range(mindepth, maxdepth):
        print(f"{depth=}")
        for variable in variables_withouth_zos:
            gnet[variable].append(
                get_glonet_rmse_for_given_days(
                    depth,
                    variable,
                    wednesdays_2024,
                    glonet_datasets_path,
                    glorys_datasets_path,
                )
            )
        if depth < 1:
            gnet["zos"].append(
                get_glonet_rmse_for_given_days(
                    depth,
                    "zos",
                    wednesdays_2024,
                    glonet_datasets_path,
                    glorys_datasets_path,
                )
            )
    return numpy.array(gnet)
