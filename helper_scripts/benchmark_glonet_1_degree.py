# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from time import perf_counter
import json

import numpy
import xarray


GLONET_SAMPLE_FIRST_DAY_DATETIMES = (
    datetime.fromisoformat("2024-01-03"),
    datetime.fromisoformat("2024-01-10"),
)
GLONET_FULL_YEAR_FIRST_DAY_DATETIMES = tuple(
    datetime.fromisoformat("2024-01-03") + timedelta(days=7 * index) for index in range(52)
)
GLONET_URL_TEMPLATE = "https://minio.dive.edito.eu/project-oceanbench/public/glonet_full_2024/{date}.zarr"
LEAD_DAYS_COUNT = 10


@dataclass(frozen=True)
class BenchmarkResult:
    scenario: str
    forecast_count: int
    open_seconds: float
    interpolate_graph_seconds: float
    compute_seconds: float | None
    total_seconds: float
    input_sizes: dict[str, int]
    output_sizes: dict[str, int]
    output_coordinates: dict[str, list[float]]
    reduction_summary: dict[str, float] | None


def _log(message: str) -> None:
    print(message, flush=True)


def _glonet_dataset_path(first_day_datetime: datetime) -> str:
    return GLONET_URL_TEMPLATE.format(date=first_day_datetime.strftime("%Y%m%d"))


def _open_glonet_dataset(first_day_datetimes: tuple[datetime, ...]) -> xarray.Dataset:
    return xarray.open_mfdataset(
        list(map(_glonet_dataset_path, first_day_datetimes)),
        engine="zarr",
        preprocess=lambda dataset: dataset.rename({"time": "lead_day_index"}).assign(
            {"lead_day_index": range(LEAD_DAYS_COUNT)}
        ),
        combine="nested",
        concat_dim="first_day_datetime",
        parallel=True,
    ).assign({"first_day_datetime": list(first_day_datetimes)})


def _interpolate_glonet_1_degree(data: xarray.Dataset) -> xarray.Dataset:
    latitude_minimum = data["lat"].min().values
    latitude_maximum = data["lat"].max().values
    longitude_minimum = data["lon"].min().values
    longitude_maximum = data["lon"].max().values

    new_latitude = numpy.arange(
        numpy.ceil(latitude_minimum - 0.5) + 0.5,
        numpy.floor(latitude_maximum + 0.5) - 0.5 + 1,
        1.0,
    )
    new_longitude = numpy.arange(
        numpy.ceil(longitude_minimum - 0.5) + 0.5,
        numpy.floor(longitude_maximum + 0.5) - 0.5 + 1,
        1.0,
    )

    data = data.chunk({"lat": -1, "lon": -1, "depth": 1})

    return data.interp(lat=new_latitude, lon=new_longitude)


def _scalarize_dataset(dataset: xarray.Dataset) -> dict[str, float]:
    return {variable_label: float(dataset[variable_label].values) for variable_label in dataset.data_vars}


def _benchmark_scenario(
    scenario: str,
    first_day_datetimes: tuple[datetime, ...],
    compute: bool,
) -> BenchmarkResult:
    _log(f"[{scenario}] opening {len(first_day_datetimes)} forecast(s)")
    total_start = perf_counter()

    open_start = perf_counter()
    input_dataset = _open_glonet_dataset(first_day_datetimes)
    open_seconds = perf_counter() - open_start
    _log(f"[{scenario}] open completed in {open_seconds:.2f}s")

    _log(f"[{scenario}] building interpolation graph")
    interpolation_start = perf_counter()
    interpolated_dataset = _interpolate_glonet_1_degree(input_dataset)
    interpolate_graph_seconds = perf_counter() - interpolation_start
    _log(f"[{scenario}] interpolation graph built in {interpolate_graph_seconds:.2f}s")

    compute_seconds = None
    reduction_summary = None
    if compute:
        _log(f"[{scenario}] forcing computation through dataset mean")
        compute_start = perf_counter()
        reduction_summary = _scalarize_dataset(interpolated_dataset.mean().compute())
        compute_seconds = perf_counter() - compute_start
        _log(f"[{scenario}] compute completed in {compute_seconds:.2f}s")

    total_seconds = perf_counter() - total_start
    _log(f"[{scenario}] total benchmark completed in {total_seconds:.2f}s")

    return BenchmarkResult(
        scenario=scenario,
        forecast_count=len(first_day_datetimes),
        open_seconds=open_seconds,
        interpolate_graph_seconds=interpolate_graph_seconds,
        compute_seconds=compute_seconds,
        total_seconds=total_seconds,
        input_sizes={dimension: int(size) for dimension, size in input_dataset.sizes.items()},
        output_sizes={dimension: int(size) for dimension, size in interpolated_dataset.sizes.items()},
        output_coordinates={
            "lat": [float(interpolated_dataset["lat"].values[0]), float(interpolated_dataset["lat"].values[-1])],
            "lon": [float(interpolated_dataset["lon"].values[0]), float(interpolated_dataset["lon"].values[-1])],
        },
        reduction_summary=reduction_summary,
    )


def _scenarios_from_args(args: Namespace) -> list[tuple[str, tuple[datetime, ...]]]:
    if args.scenario == "sample":
        return [("sample", GLONET_SAMPLE_FIRST_DAY_DATETIMES)]
    if args.scenario == "full-year":
        return [("full-year", GLONET_FULL_YEAR_FIRST_DAY_DATETIMES)]
    return [
        ("sample", GLONET_SAMPLE_FIRST_DAY_DATETIMES),
        ("full-year", GLONET_FULL_YEAR_FIRST_DAY_DATETIMES),
    ]


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Benchmark GLONET interpolation to 1 degree on the fly.",
    )
    parser.add_argument(
        "--scenario",
        choices=("sample", "full-year", "both"),
        default="both",
        help="Which workload to benchmark.",
    )
    parser.add_argument(
        "--skip-compute",
        action="store_true",
        help="Only build the interpolation graph without forcing the computation.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    for scenario, first_day_datetimes in _scenarios_from_args(args):
        result = _benchmark_scenario(
            scenario=scenario,
            first_day_datetimes=first_day_datetimes,
            compute=not args.skip_compute,
        )
        print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
