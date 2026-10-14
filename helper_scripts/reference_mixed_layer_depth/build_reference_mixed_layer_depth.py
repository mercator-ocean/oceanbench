#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Build the OceanBench reference mixed layer depth stores, one zarr per weekly first day.

Each store holds the mixed layer depth of one reference (GLORYS or GLO12) at one resolution for the ten
days starting on a Wednesday first day, computed from the reference's full vertical levels down to the
first level at or below 600 m with oceanbench.core.references.mixed_layer_depth_reference
.build_reference_mixed_layer_depth_week. The twelfth and quarter degree stores come from the native
twelfth degree temperature and salinity, the quarter degree one sampled at the points of the existing
quarter degree reference grid. The one degree store comes from the existing full level one degree
reference.

Sources are zarr path templates with a {first_day} (YYYYMMDD) placeholder. A source may carry either a
time dimension or the lead_day_index dimension of an OceanBench stage, as written by --stage-dir runs.
Without a twelfth degree template the temperature and salinity are opened from Copernicus Marine, which
needs a copernicusmarine login.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
from pathlib import Path

import dask
import numpy
import pandas
import xarray

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.mixed_layer_depth import keep_mixed_layer_depth_levels
from oceanbench.core.references import glo12, glorys
from oceanbench.core.references.mixed_layer_depth_reference import (
    REFERENCE_MIXED_LAYER_DEPTH_STORE_NAME,
    build_reference_mixed_layer_depth_week,
)

logger = logging.getLogger("build_reference_mixed_layer_depth")

LEAD_DAYS_COUNT = 10
FIRST_DAYS_2024 = pandas.date_range("2024-01-03", "2024-12-25", freq="7D")
RESOLUTIONS = ("twelfth_degree", "quarter_degree", "one_degree")
COPERNICUS_MARINE_DATASETS = {
    "glorys": {
        "thetao": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "so": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    },
    "glo12": {
        "thetao": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        "so": "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
    },
}
QUARTER_DEGREE_PATHS = {"glorys": glorys._glorys_1_4_path, "glo12": glo12._glo12_1_4_path}
ONE_DEGREE_PATHS = {"glorys": glorys._glorys_1_degree_path, "glo12": glo12._glo12_1_degree_path}
TEMPERATURE_AND_SALINITY = [Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(), Variable.SEA_WATER_SALINITY.key()]


def _temperature_and_salinity_week(dataset: xarray.Dataset, first_day: pandas.Timestamp) -> xarray.Dataset:
    dataset = keep_mixed_layer_depth_levels(rename_dataset_with_standard_names(dataset)[TEMPERATURE_AND_SALINITY])
    lead_day_index = Dimension.LEAD_DAY_INDEX.key()
    if lead_day_index in dataset.dims:
        dataset = dataset.rename({lead_day_index: "time"})
    dataset = dataset.isel(time=slice(0, LEAD_DAYS_COUNT))
    times = numpy.datetime64(first_day.date(), "ns") + numpy.arange(dataset.sizes["time"]) * numpy.timedelta64(1, "D")
    return dataset.assign_coords(time=times)


def _open_zarr(template: str, first_day: pandas.Timestamp) -> xarray.Dataset:
    return xarray.open_zarr(template.format(first_day=first_day.strftime("%Y%m%d")))


def _open_copernicus_marine_week(reference: str, first_day: pandas.Timestamp) -> xarray.Dataset:
    import copernicusmarine

    datasets = [
        copernicusmarine.open_dataset(
            dataset_id=dataset_id,
            variables=[variable],
            start_datetime=first_day.strftime("%Y-%m-%dT00:00:00"),
            end_datetime=(first_day + pandas.Timedelta(days=LEAD_DAYS_COUNT - 1)).strftime("%Y-%m-%dT00:00:00"),
        )
        for variable, dataset_id in COPERNICUS_MARINE_DATASETS[reference].items()
    ]
    return xarray.merge(datasets)


def _twelfth_degree_week(args: argparse.Namespace, first_day: pandas.Timestamp) -> xarray.Dataset:
    if args.twelfth_degree_source is None:
        dataset = _open_copernicus_marine_week(args.reference, first_day)
    else:
        dataset = _open_zarr(args.twelfth_degree_source, first_day)
    return _temperature_and_salinity_week(dataset, first_day)


def _quarter_degree_grid(args: argparse.Namespace, first_day: pandas.Timestamp) -> xarray.Dataset:
    if args.quarter_degree_grid_source is None:
        return xarray.open_zarr(QUARTER_DEGREE_PATHS[args.reference](numpy.datetime64(first_day.date())))
    return _open_zarr(args.quarter_degree_grid_source, first_day)


def _one_degree_week(args: argparse.Namespace, first_day: pandas.Timestamp) -> xarray.Dataset:
    if args.one_degree_source is None:
        dataset = xarray.open_zarr(ONE_DEGREE_PATHS[args.reference](numpy.datetime64(first_day.date())))
    else:
        dataset = _open_zarr(args.one_degree_source, first_day)
    return _temperature_and_salinity_week(dataset, first_day)


def _store_path(args: argparse.Namespace, resolution: str, first_day: pandas.Timestamp) -> Path:
    store_name = REFERENCE_MIXED_LAYER_DEPTH_STORE_NAME.format(dataset_name=args.reference, resolution=resolution)
    return Path(args.output) / store_name / f"{first_day.strftime('%Y%m%d')}.zarr"


def _write(dataset: xarray.Dataset, path: Path, attributes: dict) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    shutil.rmtree(temporary_path, ignore_errors=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = dataset.assign_attrs(attributes).chunk({"time": 1, "latitude": -1, "longitude": -1})
    for variable in dataset.variables.values():
        variable.encoding.clear()
    dataset.to_zarr(temporary_path, mode="w", consolidated=True)
    temporary_path.rename(path)


def build_week(args: argparse.Namespace, first_day: pandas.Timestamp, script_sha256: str) -> None:
    pending = [
        resolution
        for resolution in args.resolutions
        if args.overwrite or not _store_path(args, resolution, first_day).exists()
    ]
    if not pending:
        logger.info("%s %s already built", args.reference, first_day.date())
        return
    attributes = {
        "title": f"{args.reference} reference mixed layer depth",
        "method": "density threshold 0.03 kg/m3 relative to 10 m on sigma0, interpolated, capped at 600 m",
        "first_day": first_day.strftime("%Y-%m-%d"),
        "builder_script_sha256": script_sha256,
    }
    twelfth_degree_pending = [resolution for resolution in pending if resolution != "one_degree"]
    if twelfth_degree_pending:
        source = _twelfth_degree_week(args, first_day)
        for resolution in twelfth_degree_pending:
            grid = _quarter_degree_grid(args, first_day) if resolution == "quarter_degree" else None
            mixed_layer_depth = build_reference_mixed_layer_depth_week(source, grid).compute()
            _write(
                mixed_layer_depth, _store_path(args, resolution, first_day), {**attributes, "resolution": resolution}
            )
            logger.info("%s %s %s written", args.reference, first_day.date(), resolution)
        del source
    if "one_degree" in pending:
        mixed_layer_depth = build_reference_mixed_layer_depth_week(_one_degree_week(args, first_day)).compute()
        _write(
            mixed_layer_depth, _store_path(args, "one_degree", first_day), {**attributes, "resolution": "one_degree"}
        )
        logger.info("%s %s one_degree written", args.reference, first_day.date())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, choices=sorted(COPERNICUS_MARINE_DATASETS))
    parser.add_argument("--output", required=True, help="output root, stores land in <output>/<store name>/")
    parser.add_argument("--first-days", nargs="+", dest="first_days", help="YYYY-MM-DD, default all 2024 weeks")
    parser.add_argument("--resolutions", nargs="+", choices=RESOLUTIONS, default=list(RESOLUTIONS))
    parser.add_argument("--twelfth-degree-source", dest="twelfth_degree_source", default=None)
    parser.add_argument("--quarter-degree-grid-source", dest="quarter_degree_grid_source", default=None)
    parser.add_argument("--one-degree-source", dest="one_degree_source", default=None)
    parser.add_argument("--workers", type=int, default=8, help="dask threads")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO", dest="log_level")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    first_days = FIRST_DAYS_2024 if args.first_days is None else pandas.to_datetime(args.first_days)
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with dask.config.set(scheduler="threads", num_workers=args.workers):
        for first_day in first_days:
            build_week(args, first_day, script_sha256)
    return 0


if __name__ == "__main__":
    sys.exit(main())
