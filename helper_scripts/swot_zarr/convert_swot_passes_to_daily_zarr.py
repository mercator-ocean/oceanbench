# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Convert the SWOT L3 KaRIn passes stored in the OceanBench bucket into one Zarr store per day.

Source: oceanbench-bucket/class4/swot/l3/<reproc|forward>/<year>/SWOT_L3_*.nc
Target: oceanbench-bucket/class4/swot/l3/zarr-v2/<year>/<YYYYMMDD>.zarr

Each daily store is a flat list of observations along the ``observations`` dimension, with
``sea_surface_height_above_geoid`` (from ``ssha_filtered``), ``latitude``, ``longitude`` on [-180, 180)
and ``time``. A day is built from the reprocessed passes when any exist, from the forward passes otherwise.
One observation out of ``--subsample-factor`` is kept, drawn at random with the date as seed,
so that rebuilding a day always keeps the same observations.

Each daily store records its basis version, the names of the pass files it was built from and its
subsample factor. A day is converted again only when one of them has changed since its last conversion,
or with ``--overwrite``. Running the script again after new passes are added
to the bucket therefore converts the new days, the days completed by a new pass across midnight,
and the days whose forward passes have been replaced by reprocessed ones.

Credentials are read from CF_KEY and CF_SECRET, or from AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.
"""

import argparse
import collections
import datetime
import json
import logging
import os
import re

import numpy
import pandas
import s3fs
import xarray

BUCKET_NAME = "oceanbench-bucket"
SOURCE_PREFIX = "class4/swot/l3"
TARGET_PREFIX = "class4/swot/l3/zarr-v2"
PROCESSING_MODES_BY_PRIORITY = ("reproc", "forward")
SOURCE_SEA_SURFACE_HEIGHT_VARIABLE = "ssha_filtered"
SEA_SURFACE_HEIGHT_VARIABLE = "sea_surface_height_above_geoid"
LATITUDE_VARIABLE = "latitude"
LONGITUDE_VARIABLE = "longitude"
TIME_VARIABLE = "time"
OBSERVATION_DIMENSION = "observations"
OBSERVATION_BASIS_VERSION = "v2.1"
DEFAULT_SUBSAMPLE_FACTOR = 10
CLOUDFERRO_ENDPOINT = "https://s3.waw3-1.cloudferro.com"
CLOUDFERRO_REGION = "waw3-1"
PASS_DATETIME_PATTERN = re.compile(r"(\d{8})T\d{6}")
SINGLE_PRECISION_ENCODING = {"dtype": "float32"}
PACKING_ENCODING_KEYS = ("dtype", "scale_factor", "add_offset", "_FillValue")
TIME_ENCODING_KEYS = ("units", "calendar", "dtype")

logger = logging.getLogger("convert_swot_passes_to_daily_zarr")


def _storage_options() -> dict:
    key = os.environ.get("CF_KEY") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("CF_SECRET") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        raise SystemExit(
            "Missing credentials: set CF_KEY and CF_SECRET, or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
        )
    return {
        "key": key,
        "secret": secret,
        "client_kwargs": {"endpoint_url": CLOUDFERRO_ENDPOINT, "region_name": CLOUDFERRO_REGION},
        "config_kwargs": {
            "s3": {"addressing_style": "path"},
            "max_pool_connections": 64,
            "retries": {"max_attempts": 10},
        },
    }


def _days_covered_by_pass(pass_path: str) -> frozenset[datetime.date]:
    pass_datetimes = PASS_DATETIME_PATTERN.findall(os.path.basename(pass_path))
    if not pass_datetimes:
        raise ValueError(f"No date found in the pass file name: {pass_path}")
    return frozenset(pandas.date_range(pass_datetimes[0], pass_datetimes[-1], freq="D").date)


def _days_covered_by_pass_path(file_system: s3fs.S3FileSystem, processing_mode: str) -> dict:
    pass_paths = file_system.find(f"{BUCKET_NAME}/{SOURCE_PREFIX}/{processing_mode}")
    return {
        pass_path: _days_covered_by_pass(pass_path) for pass_path in sorted(pass_paths) if pass_path.endswith(".nc")
    }


def _processing_mode_and_pass_paths_of_day(
    day: datetime.date,
    days_covered_by_pass_path_by_processing_mode: dict,
) -> tuple[str | None, list[str]]:
    for processing_mode in PROCESSING_MODES_BY_PRIORITY:
        pass_paths = [
            pass_path
            for pass_path, covered_days in days_covered_by_pass_path_by_processing_mode[processing_mode].items()
            if day in covered_days
        ]
        if pass_paths:
            return processing_mode, pass_paths
    return None, []


def _longitude_on_minus_180_to_180(longitude: xarray.DataArray) -> xarray.DataArray:
    is_outside_range = (longitude < -180.0) | (longitude >= 180.0)
    return xarray.where(is_outside_range, ((longitude + 180.0) % 360.0) - 180.0, longitude)


def _flattened_values(pass_dataset: xarray.Dataset, variable_name: str) -> numpy.ndarray:
    sea_surface_height = pass_dataset[SOURCE_SEA_SURFACE_HEIGHT_VARIABLE]
    broadcast_variable = pass_dataset[variable_name].broadcast_like(sea_surface_height)
    return broadcast_variable.transpose(*sea_surface_height.dims).values.ravel()


def _observations_of_pass(file_system: s3fs.S3FileSystem, pass_path: str) -> xarray.Dataset:
    source_variables = [SOURCE_SEA_SURFACE_HEIGHT_VARIABLE, LATITUDE_VARIABLE, LONGITUDE_VARIABLE, TIME_VARIABLE]
    with file_system.open(pass_path, mode="rb") as pass_file:
        pass_dataset = xarray.open_dataset(pass_file, engine="h5netcdf")[source_variables].load()
    target_variable_by_source_variable = {
        SOURCE_SEA_SURFACE_HEIGHT_VARIABLE: SEA_SURFACE_HEIGHT_VARIABLE,
        LATITUDE_VARIABLE: LATITUDE_VARIABLE,
        LONGITUDE_VARIABLE: LONGITUDE_VARIABLE,
        TIME_VARIABLE: TIME_VARIABLE,
    }
    observations = xarray.Dataset(
        {
            target_variable: (OBSERVATION_DIMENSION, _flattened_values(pass_dataset, source_variable))
            for source_variable, target_variable in target_variable_by_source_variable.items()
        }
    )
    is_valid_observation = observations.notnull().to_array().all("variable")
    return observations.isel({OBSERVATION_DIMENSION: numpy.flatnonzero(is_valid_observation.values)})


def _observations_of_day(file_system: s3fs.S3FileSystem, day: datetime.date, pass_paths: list[str]) -> xarray.Dataset:
    observations_of_passes = xarray.concat(
        [_observations_of_pass(file_system, pass_path) for pass_path in pass_paths],
        dim=OBSERVATION_DIMENSION,
    )
    day_start = numpy.datetime64(day, "ns")
    day_end = day_start + numpy.timedelta64(1, "D")
    observation_times = observations_of_passes[TIME_VARIABLE].values
    is_within_day = (observation_times >= day_start) & (observation_times < day_end)
    observations_within_day = observations_of_passes.isel({OBSERVATION_DIMENSION: numpy.flatnonzero(is_within_day)})
    return observations_within_day.assign(
        {LONGITUDE_VARIABLE: _longitude_on_minus_180_to_180(observations_within_day[LONGITUDE_VARIABLE])}
    )


def _subsampled_observations(observations: xarray.Dataset, day: datetime.date, subsample_factor: int) -> xarray.Dataset:
    random_generator = numpy.random.default_rng(seed=int(day.strftime("%Y%m%d")))
    is_kept_observation = random_generator.random(observations.sizes[OBSERVATION_DIMENSION]) < 1.0 / subsample_factor
    return observations.isel({OBSERVATION_DIMENSION: numpy.flatnonzero(is_kept_observation)})


def _packed_or_single_precision_encoding(source_encoding: dict) -> dict:
    is_packed_as_signed_integers = (
        "scale_factor" in source_encoding and numpy.dtype(source_encoding.get("dtype", "float64")).kind == "i"
    )
    if is_packed_as_signed_integers:
        return {key: source_encoding[key] for key in PACKING_ENCODING_KEYS if key in source_encoding}
    return SINGLE_PRECISION_ENCODING


def _time_encoding(source_encoding: dict) -> dict:
    return {key: source_encoding[key] for key in TIME_ENCODING_KEYS if key in source_encoding}


def _target_encoding(file_system: s3fs.S3FileSystem, pass_path: str) -> dict:
    with file_system.open(pass_path, mode="rb") as pass_file:
        pass_dataset = xarray.open_dataset(pass_file, engine="h5netcdf")
        return {
            SEA_SURFACE_HEIGHT_VARIABLE: _packed_or_single_precision_encoding(
                pass_dataset[SOURCE_SEA_SURFACE_HEIGHT_VARIABLE].encoding
            ),
            LATITUDE_VARIABLE: _packed_or_single_precision_encoding(pass_dataset[LATITUDE_VARIABLE].encoding),
            LONGITUDE_VARIABLE: _packed_or_single_precision_encoding(pass_dataset[LONGITUDE_VARIABLE].encoding),
            TIME_VARIABLE: _time_encoding(pass_dataset[TIME_VARIABLE].encoding),
        }


def _with_metadata(
    observations: xarray.Dataset,
    processing_mode: str,
    pass_file_names: list[str],
    subsample_factor: int,
) -> xarray.Dataset:
    return observations.assign_attrs(
        obs_basis_version=OBSERVATION_BASIS_VERSION,
        source_variable=SOURCE_SEA_SURFACE_HEIGHT_VARIABLE,
        processing_mode=processing_mode,
        source_pass_files=pass_file_names,
        subsample_factor=subsample_factor,
    ).assign(
        {
            SEA_SURFACE_HEIGHT_VARIABLE: observations[SEA_SURFACE_HEIGHT_VARIABLE].assign_attrs(
                standard_name=SEA_SURFACE_HEIGHT_VARIABLE, units="m"
            ),
            LATITUDE_VARIABLE: observations[LATITUDE_VARIABLE].assign_attrs(
                standard_name="latitude", units="degrees_north"
            ),
            LONGITUDE_VARIABLE: observations[LONGITUDE_VARIABLE].assign_attrs(
                standard_name="longitude", units="degrees_east"
            ),
            TIME_VARIABLE: observations[TIME_VARIABLE].assign_attrs(standard_name="time"),
        }
    )


def _target_path(day: datetime.date) -> str:
    return f"{BUCKET_NAME}/{TARGET_PREFIX}/{day:%Y}/{day:%Y%m%d}.zarr"


def _pass_file_names(pass_paths: list[str]) -> list[str]:
    return sorted(os.path.basename(pass_path) for pass_path in pass_paths)


def _converted_day_attributes(file_system: s3fs.S3FileSystem, target_path: str) -> dict:
    try:
        consolidated_metadata = json.loads(file_system.cat_file(f"{target_path}/.zmetadata"))
    except FileNotFoundError:
        return {}
    return consolidated_metadata["metadata"][".zattrs"]


def _is_up_to_date(
    file_system: s3fs.S3FileSystem,
    target_path: str,
    pass_file_names: list[str],
    subsample_factor: int,
) -> bool:
    converted_day_attributes = _converted_day_attributes(file_system, target_path)
    return (
        converted_day_attributes.get("obs_basis_version") == OBSERVATION_BASIS_VERSION
        and converted_day_attributes.get("source_pass_files") == pass_file_names
        and converted_day_attributes.get("subsample_factor") == subsample_factor
    )


def _convert_day(
    file_system: s3fs.S3FileSystem,
    day: datetime.date,
    days_covered_by_pass_path_by_processing_mode: dict,
    subsample_factor: int,
    overwrite: bool,
) -> str:
    target_path = _target_path(day)
    processing_mode, pass_paths = _processing_mode_and_pass_paths_of_day(
        day, days_covered_by_pass_path_by_processing_mode
    )
    if not pass_paths:
        return "no pass"
    pass_file_names = _pass_file_names(pass_paths)
    if not overwrite and _is_up_to_date(file_system, target_path, pass_file_names, subsample_factor):
        return "already up to date"
    observations = _subsampled_observations(_observations_of_day(file_system, day, pass_paths), day, subsample_factor)
    if observations.sizes[OBSERVATION_DIMENSION] == 0:
        return "no observation"
    _with_metadata(observations, processing_mode, pass_file_names, subsample_factor).to_zarr(
        store=file_system.get_mapper(target_path),
        mode="w",
        consolidated=True,
        encoding=_target_encoding(file_system, pass_paths[0]),
    )
    logger.info(
        "%s: %d observations from %d %s passes",
        day,
        observations.sizes[OBSERVATION_DIMENSION],
        len(pass_paths),
        processing_mode,
    )
    return "converted"


def _convert_day_or_report_failure(
    file_system: s3fs.S3FileSystem,
    day: datetime.date,
    days_covered_by_pass_path_by_processing_mode: dict,
    subsample_factor: int,
    overwrite: bool,
) -> str:
    try:
        return _convert_day(file_system, day, days_covered_by_pass_path_by_processing_mode, subsample_factor, overwrite)
    except Exception:
        logger.exception("%s: conversion failed, the day is not written", day)
        return "failed"


def _all_covered_days(days_covered_by_pass_path_by_processing_mode: dict) -> frozenset[datetime.date]:
    return frozenset().union(
        *(
            covered_days
            for days_covered_by_pass_path in days_covered_by_pass_path_by_processing_mode.values()
            for covered_days in days_covered_by_pass_path.values()
        )
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--start", type=datetime.date.fromisoformat, help="first day, YYYY-MM-DD, default: first day covered by a pass"
    )
    parser.add_argument(
        "--end",
        type=datetime.date.fromisoformat,
        help="last day included, YYYY-MM-DD, default: last day covered by a pass",
    )
    parser.add_argument("--subsample-factor", type=int, default=DEFAULT_SUBSAMPLE_FACTOR, dest="subsample_factor")
    parser.add_argument("--overwrite", action="store_true", help="convert again the days that are already up to date")
    parser.add_argument("--log-level", default="INFO", dest="log_level")
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    logging.basicConfig(level=arguments.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")
    file_system = s3fs.S3FileSystem(**_storage_options())
    days_covered_by_pass_path_by_processing_mode = {
        processing_mode: _days_covered_by_pass_path(file_system, processing_mode)
        for processing_mode in PROCESSING_MODES_BY_PRIORITY
    }
    all_covered_days = _all_covered_days(days_covered_by_pass_path_by_processing_mode)
    if not all_covered_days:
        raise SystemExit(f"No SWOT pass found under {BUCKET_NAME}/{SOURCE_PREFIX}.")
    first_day = arguments.start or min(all_covered_days)
    last_day = arguments.end or max(all_covered_days)
    logger.info("Converting the days from %s to %s", first_day, last_day)
    days = pandas.date_range(first_day, last_day, freq="D").date
    statuses = collections.Counter(
        _convert_day_or_report_failure(
            file_system,
            day,
            days_covered_by_pass_path_by_processing_mode,
            arguments.subsample_factor,
            arguments.overwrite,
        )
        for day in days
    )
    logger.info("Summary: %s", dict(statuses))


if __name__ == "__main__":
    main()
