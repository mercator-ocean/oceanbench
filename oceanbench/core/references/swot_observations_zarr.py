# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
SWOT l3 observations (zarr files - OceanBench bucket) as Class IV
observations, in the 1D format of the other altimetry references.

Same design as oceanbench.core.references.observations : only the acquisition
days matching the challenger runs are opened (one zarr store per day, every 
observation point is tagged with the first_day_datetime of the run(s) it matches, 
and the selected observations can be staged locally so that challengers with 
the same runs share one download.)

Remote access reuses oceanbench.core.remote_http : transient errors are retried
with exponential backoff, and a network error is never read as an absent day.

SEA_SURFACE_HEIGHT_ABOVE_GEOID_KEY is fed by the SWOT L3 variable stored by the
pre-processing script (conversion_swot_netcdf_to_zarr_bucketCF.py).
"""

#==============================================================================

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aiohttp
import numpy
import pandas
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.local_stage import (
    local_stage_directory,
    open_or_create_local_stage_dataset,
    should_stage_locally,
    write_dataset_to_local_stage,
)
from oceanbench.core.references.observations import _forecast_observation_matches
from oceanbench.core.remote_http import (
    RetryingHTTPFileSystem,
    open_remote_zarr,
    remote_store_exists,
)

#==============================================================================

logger = logging.getLogger(__name__)

SEA_SURFACE_HEIGHT_ABOVE_GEOID_KEY = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
TIME_KEY = Dimension.TIME.key()
LATITUDE_KEY = Dimension.LATITUDE.key()
LONGITUDE_KEY = Dimension.LONGITUDE.key()
DEPTH_KEY = Dimension.DEPTH.key()
FIRST_DAY_DATETIME_KEY = Dimension.FIRST_DAY_DATETIME.key()
LEAD_DAY_INDEX_KEY = Dimension.LEAD_DAY_INDEX.key()

SWOT_HTTP_PREFIX = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/class4/swot/l3/zarr"
SWOT_SUBFOLDERS = ["reproc", "forward"]

SWOT_BASIS_VERSION_ATTRIBUTE = "obs_basis_version"
EXPECTED_SWOT_BASIS_VERSION = "v1.0"  # version written by the SWOT L3 pre-processing

LOCAL_STAGE_SWOT_KEY = "swot-observations"
SWOT_STAGE_VERSION = "v2"

EARTH_RADIUS_KM = 6378.1
KM_PER_DEGREE = 111.317

SWOT_STORAGE_OPTIONS = {
    "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=600, connect=30, sock_read=120)},
}
_HTTP_FILE_SYSTEM = RetryingHTTPFileSystem(**SWOT_STORAGE_OPTIONS)


#==============================================================================
# HELPERS

def _swot_day_url(day_string: str) -> str | None:
    # First existing zarr url for that day (reproc has priority over forward).
    # remote_store_exists lets network errors propagate, after the retries.
    for subfolder in SWOT_SUBFOLDERS:
        url = f"{SWOT_HTTP_PREFIX}/{subfolder}/{day_string[:4]}/{day_string}.zarr"
        if remote_store_exists(_HTTP_FILE_SYSTEM, url):
            return url
    return None


def _require_swot_basis_version(day_dataset: xarray.Dataset) -> None:
    # Strict check, like _require_observation_basis_version in observations.py.
    found_basis_version = day_dataset.attrs.get(SWOT_BASIS_VERSION_ATTRIBUTE)
    if found_basis_version != EXPECTED_SWOT_BASIS_VERSION:
        raise ValueError(
            "SWOT day store was built on an unexpected basis version. "
            f"Expected {SWOT_BASIS_VERSION_ATTRIBUTE}={EXPECTED_SWOT_BASIS_VERSION!r}, "
            f"found {found_basis_version!r}."
        )


def _median_along_track_spacing_degrees(latitude: numpy.ndarray, longitude: numpy.ndarray) -> float | None:
    # Median native along-track spacing (degrees). The median is used because
    # it is robust to the much larger spacing at the gaps of the track
    # (cycle and calval boundaries) : those must not inflate the estimate.
    if latitude.size < 2:
        return None
    latitude_radians_first = numpy.radians(latitude[:-1])
    longitude_radians_first = numpy.radians(longitude[:-1])
    latitude_radians_second = numpy.radians(latitude[1:])
    longitude_radians_second = numpy.radians(longitude[1:])
    haversine = (
        numpy.sin((latitude_radians_second - latitude_radians_first) / 2) ** 2
        + numpy.cos(latitude_radians_first)
        * numpy.cos(latitude_radians_second)
        * numpy.sin((longitude_radians_second - longitude_radians_first) / 2) ** 2
    )
    segment_km = 2 * EARTH_RADIUS_KM * numpy.arcsin(numpy.sqrt(haversine))
    segment_km = segment_km[segment_km > 0]
    if segment_km.size == 0:
        return None
    return float(numpy.median(segment_km)) / KM_PER_DEGREE


def _subsample_mask(day: pandas.Timestamp, observation_count: int, subsample_step: int) -> numpy.ndarray:
    # Bernoulli mask with a per-day fixed seed. A regular stride, even with a
    # random offset, only ever samples a few transverse positions of the swath.
    if subsample_step <= 1:
        return numpy.ones(observation_count, dtype=bool)
    generator = numpy.random.default_rng(int(day.strftime("%Y%m%d")))
    return generator.random(observation_count) < 1.0 / subsample_step


def _swot_stage_path(
    first_day_timestamps: pandas.DatetimeIndex,
    lead_days_count: int,
    subsample: int,
    observation_period: tuple[pandas.Timestamp, pandas.Timestamp] | None,
) -> Path:
    # Stage path of the selected observations : it depends only on the runs,
    # the number of lead days, the subsample and the observation period, so
    # every challenger with the same request reuses the same staged dataset.
    payload = ",".join(first_day_timestamps.sort_values().strftime("%Y-%m-%d"))
    if observation_period is not None:
        payload += f",{observation_period[0]:%Y-%m-%d},{observation_period[1]:%Y-%m-%d}"
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()[:12]
    return local_stage_directory() / (
        f"swot-{SWOT_STAGE_VERSION}-{int(first_day_timestamps.min().strftime('%Y%m%d'))}-"
        f"{int(first_day_timestamps.max().strftime('%Y%m%d'))}-{lead_days_count}d-{subsample}-{digest}.zarr"
    )



#==============================================================================

def swot_observations(
    challenger_dataset: xarray.Dataset,
    subsample: int = 1,
    observation_period: tuple[pandas.Timestamp, pandas.Timestamp] | None = None,
) -> xarray.Dataset:
    """Return the SWOT observations matching the runs of the challenger
    (Class IV validation scheme) : one line per observation point, every point
    tagged with the first_day_datetime of the run(s) it matches.

    subsample keeps on average 1 point over N with a per-day Bernoulli mask
    (fixed seed) : a regular stride, even with an offset, would alias with the
    transverse geometry of the swath. It reduces the memory footprint, not the
    download : every selected day store is always loaded in full.

    observation_period restricts the downloaded acquisition days to that
    period : the pointwise outputs of a single date download one day instead
    of the full windows of the runs. The RMSD table needs the full windows and
    leaves it None.
    """

    first_day_datetimes = challenger_dataset[FIRST_DAY_DATETIME_KEY].values
    first_day_timestamps = pandas.to_datetime(first_day_datetimes)
    lead_days_count = challenger_dataset.sizes[LEAD_DAY_INDEX_KEY]

    def _open_staged_swot(stage_path: Path) -> xarray.Dataset:
        return xarray.open_dataset(stage_path, engine="zarr")

    def _load_day(day: pandas.Timestamp) -> tuple[xarray.Dataset | None, float | None]:
        url = _swot_day_url(day.strftime("%Y%m%d"))
        if url is None:
            return None, None
        day_full = open_remote_zarr(
            url,
            storage_options=SWOT_STORAGE_OPTIONS,
            consolidated=True,
            decode_cf=False,
        )
        _require_swot_basis_version(day_full)
        day_full = day_full[[SEA_SURFACE_HEIGHT_ABOVE_GEOID_KEY, TIME_KEY, LATITUDE_KEY, LONGITUDE_KEY]].load()
        day_datetimes = pandas.to_datetime(day_full[TIME_KEY].values)
        if not bool((day_datetimes.normalize() == day.normalize()).all()):
            raise ValueError(
                f"SWOT day store {day:%Y-%m-%d} contains times outside that day : inconsistent zarr."
            )
        spacing_day = _median_along_track_spacing_degrees(
            numpy.asarray(day_full[LATITUDE_KEY].values, dtype="float64"),
            numpy.asarray(day_full[LONGITUDE_KEY].values, dtype="float64"),
        )
        subsample_step = max(int(subsample), 1)
        day_subsampled = day_full.isel(
            observations=_subsample_mask(day, day_full.sizes["observations"], subsample_step)
        )
        return day_subsampled, spacing_day

    def _build_swot_observations() -> xarray.Dataset:
        start = first_day_timestamps.min().normalize()
        end = first_day_timestamps.max().normalize() + pandas.Timedelta(days=lead_days_count)
        days = pandas.date_range(start, end - pandas.Timedelta(days=1), freq="D")
        if observation_period is not None:
            period_start, period_end = observation_period
            days = days[(days >= period_start.normalize()) & (days <= period_end.normalize())]
            if days.empty:
                raise ValueError("The requested observation period does not intersect the runs of the model.")

        with ThreadPoolExecutor(max_workers=min(6, len(days))) as executor:
            day_datasets = list(executor.map(_load_day, days))

        day_parts = [day_dataset for day_dataset, _ in day_datasets if day_dataset is not None]
        day_spacings = [spacing for _, spacing in day_datasets if spacing is not None]
        missing_days_count = len(day_datasets) - len(day_parts)

        if not day_parts:
            raise RuntimeError("No SWOT zarr store was found for the requested days.")
        if missing_days_count:
            logger.info(f"{missing_days_count} day(s) without SWOT zarr store : skipped")

        combined = xarray.concat(day_parts, dim="observations")
        source_attributes = dict(day_parts[0].attrs)
        native_resolution_degrees = float(numpy.median(day_spacings)) if day_spacings else None
        del day_parts, day_datasets

        ssh_values = numpy.asarray(combined[SEA_SURFACE_HEIGHT_ABOVE_GEOID_KEY].values, dtype="float64")
        latitude_values = numpy.asarray(combined[LATITUDE_KEY].values, dtype="float64")
        longitude_values = numpy.asarray(combined[LONGITUDE_KEY].values, dtype="float64")
        observation_datetimes = pandas.DatetimeIndex(pandas.to_datetime(combined[TIME_KEY].values))
        del combined

        selected_observation_indices, selected_run_indices = _forecast_observation_matches(
            observation_datetimes, first_day_timestamps, lead_days_count
        )
        if selected_observation_indices.size == 0:
            raise RuntimeError("No SWOT observation matches the runs of the model.")

        observations_dataset = xarray.Dataset(
            {
                SEA_SURFACE_HEIGHT_ABOVE_GEOID_KEY: (
                    ("observations",),
                    ssh_values[selected_observation_indices],
                )
            },
            coords={
                TIME_KEY: (
                    ("observations",),
                    observation_datetimes[selected_observation_indices].to_numpy(),
                ),
                LATITUDE_KEY: (
                    ("observations",),
                    latitude_values[selected_observation_indices],
                ),
                LONGITUDE_KEY: (
                    ("observations",),
                    longitude_values[selected_observation_indices],
                ),
                DEPTH_KEY: (
                    ("observations",),
                    numpy.zeros(selected_observation_indices.size),
                ),  # surface
                FIRST_DAY_DATETIME_KEY: (
                    ("observations",),
                    first_day_datetimes[selected_run_indices],
                ),
            },
        )

        observations_dataset.attrs.update(source_attributes)
        if native_resolution_degrees is not None:
            observations_dataset.attrs["obs_along_track_resolution_degrees"] = native_resolution_degrees
        observations_dataset.attrs["subsample"] = max(int(subsample), 1)
        for key in (SEA_SURFACE_HEIGHT_ABOVE_GEOID_KEY, TIME_KEY, LATITUDE_KEY, LONGITUDE_KEY):
            observations_dataset[key].attrs["standard_name"] = key

        logger.info(
            f"SWOT : {observations_dataset.sizes['observations']:,} points, "
            f"{numpy.unique(observations_dataset[FIRST_DAY_DATETIME_KEY].values).size} runs covered"
            + (
                f", along swath resolution ≈ {native_resolution_degrees:.4f}°"
                if native_resolution_degrees is not None
                else ""
            )
        )
        return observations_dataset

    def _build_stage(stage_path: Path) -> None:
        write_dataset_to_local_stage(
            _build_swot_observations(),
            stage_path,
            load_before_write=True,
            clear_chunk_encoding=True,
        )

    if should_stage_locally(LOCAL_STAGE_SWOT_KEY):
        return open_or_create_local_stage_dataset(
            _swot_stage_path(first_day_timestamps, lead_days_count, subsample, observation_period),
            open_staged_dataset=_open_staged_swot,
            build_stage=_build_stage,
        )
    return _build_swot_observations()