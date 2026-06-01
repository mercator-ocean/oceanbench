# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
from pathlib import Path
import sys

import numpy
import xarray

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from oceanbench.core.dataset_utils import Dimension, LEAD_DAYS_COUNT, Variable

DEFAULT_FIRST_DAY = "2026-05-13"
DEFAULT_OUTPUT_DIRECTORY = Path("dev/live-evaluations/synthetic-buckets")


def _coordinate_dataarrays() -> dict[str, xarray.DataArray]:
    return {
        "time": xarray.DataArray(numpy.arange(LEAD_DAYS_COUNT, dtype=numpy.int16), dims=("time",)),
        Dimension.DEPTH.key(): xarray.DataArray(
            numpy.array([0.5, 10.0, 30.0, 50.0, 100.0, 300.0, 1000.0], dtype=numpy.float32),
            dims=(Dimension.DEPTH.key(),),
        ),
        Dimension.LATITUDE.key(): xarray.DataArray(
            numpy.arange(-80.0, 91.0, 1.0, dtype=numpy.float32),
            dims=(Dimension.LATITUDE.key(),),
        ),
        Dimension.LONGITUDE.key(): xarray.DataArray(
            numpy.arange(-180.0, 180.0, 1.0, dtype=numpy.float32),
            dims=(Dimension.LONGITUDE.key(),),
        ),
    }


def _synthetic_dataset(offset: float) -> xarray.Dataset:
    coords = _coordinate_dataarrays()
    lead = coords["time"].astype(numpy.float32)
    depth = coords[Dimension.DEPTH.key()]
    latitude = coords[Dimension.LATITUDE.key()]
    longitude = coords[Dimension.LONGITUDE.key()]
    latitude_radian = numpy.deg2rad(latitude)
    longitude_radian = numpy.deg2rad(longitude)

    surface_pattern = (
        0.12 * xarray.ufuncs.sin(latitude_radian) + 0.04 * xarray.ufuncs.cos(longitude_radian) + 0.01 * lead + offset
    )
    temperature = (
        18.0
        - 0.018 * depth
        + 1.8 * xarray.ufuncs.cos(latitude_radian)
        + 0.2 * xarray.ufuncs.sin(longitude_radian)
        - 0.03 * lead
        + offset
    )
    salinity = (
        35.0
        + 0.0015 * depth
        + 0.08 * xarray.ufuncs.sin(latitude_radian)
        + 0.02 * xarray.ufuncs.cos(longitude_radian)
        + 0.01 * lead
        + offset
    )
    eastward_current = (
        0.12 * xarray.ufuncs.cos(latitude_radian)
        + 0.02 * xarray.ufuncs.sin(longitude_radian)
        - 0.00004 * depth
        + 0.004 * lead
        + offset
    )
    northward_current = (
        0.03 * xarray.ufuncs.cos(latitude_radian)
        + 0.08 * xarray.ufuncs.sin(longitude_radian)
        - 0.00003 * depth
        + 0.003 * lead
        + offset
    )

    dataset = xarray.Dataset(
        data_vars={
            "zos": (
                ("time", Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()),
                surface_pattern.transpose("time", Dimension.LATITUDE.key(), Dimension.LONGITUDE.key())
                .astype(numpy.float32)
                .values,
                {"standard_name": Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()},
            ),
            "thetao": (
                ("time", Dimension.DEPTH.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()),
                temperature.transpose(
                    "time",
                    Dimension.DEPTH.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                )
                .astype(numpy.float32)
                .values,
                {"standard_name": Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()},
            ),
            "so": (
                ("time", Dimension.DEPTH.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()),
                salinity.transpose(
                    "time",
                    Dimension.DEPTH.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                )
                .astype(numpy.float32)
                .values,
                {"standard_name": Variable.SEA_WATER_SALINITY.key()},
            ),
            "uo": (
                ("time", Dimension.DEPTH.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()),
                eastward_current.transpose(
                    "time",
                    Dimension.DEPTH.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                )
                .astype(numpy.float32)
                .values,
                {"standard_name": Variable.EASTWARD_SEA_WATER_VELOCITY.key()},
            ),
            "vo": (
                ("time", Dimension.DEPTH.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()),
                northward_current.transpose(
                    "time",
                    Dimension.DEPTH.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                )
                .astype(numpy.float32)
                .values,
                {"standard_name": Variable.NORTHWARD_SEA_WATER_VELOCITY.key()},
            ),
        },
        coords=coords,
        attrs={"title": "Synthetic OceanBench live-evaluation development dataset"},
    )
    return dataset.chunk({"time": 1, Dimension.DEPTH.key(): 1, Dimension.LATITUDE.key(): 60})


def _write_zarr(dataset: xarray.Dataset, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(path, consolidated=True)


def create_synthetic_live_evaluation_buckets(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    first_day: str = DEFAULT_FIRST_DAY,
) -> dict[str, Path]:
    datetime.fromisoformat(first_day)
    glonet_path = output_directory / "glonet" / first_day / f"{first_day}.zarr"
    glo12_path = output_directory / "glo12" / first_day / f"{first_day}.zarr"
    _write_zarr(_synthetic_dataset(offset=0.02), glonet_path)
    _write_zarr(_synthetic_dataset(offset=0.0), glo12_path)
    return {
        "glonet": glonet_path,
        "glo12": glo12_path,
    }


def main() -> None:
    paths = create_synthetic_live_evaluation_buckets()
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
