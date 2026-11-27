# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Write the files the viewer streams for one evaluated challenger and region.

The Class IV files are derived from the match-ups the scores were computed from, so the viewer
shows the same population as the scores file. Pooled RMSD values come from
``compute_class4_rmsd_table``, the function behind the official Class IV scores. Bias, the error
geography and the confidence intervals have no official counterpart and are computed here.
"""

from dataclasses import dataclass
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

import numpy
import pandas
import pyarrow
import pyarrow.compute
import pyarrow.parquet
import xarray

from oceanbench.core import eddy_census
from oceanbench.core.classIV_support import compute_class4_rmsd_table
from oceanbench.core.classIV_matchups import Class4Matchups, class4_matchups
from oceanbench.core.dataset_source import get_challenger_description, get_challenger_kind
from oceanbench.core.dataset_utils import DEPTH_BIN_DISPLAY_ORDER, Dimension, Variable
from oceanbench.core.regions import GLOBAL_REGION_NAME, RegionSpec, padded_region, subset_dataset_to_region
from oceanbench.core.schema_validation import validate_against_schema
from oceanbench.core.version import __version__
from oceanbench.publish.column_store import COLUMN_STORE_SUFFIX, build_column_store
from oceanbench.pyramids import build_pyramid, viewer_layers

MATCHUP_PARQUET_FILENAME = "class4-matchups.parquet"
EDDY_CENSUS_FILENAME = "eddies.json"
YEAR_ERROR_GEOGRAPHY_FILENAME = "year-error-geography.json"
YEAR_RMSD_BY_START_FILENAME = "year-rmsd-by-start.json"
RMSD_BY_DEPTH_FILENAME = "rmsd-by-depth.json"
PYRAMID_SUFFIX = ".zarr"
PYRAMID_MANIFEST_SUFFIX = ".viewer-manifest.json"

# The viewer reads the match-ups by row group. A row group holds whole (variable, depth_bin) blocks of a
# single (start_date, lead_day) pair, packed up to TARGET_ROW_GROUP_ROWS rows to keep the footer small.
TARGET_ROW_GROUP_ROWS = 400_000
MAXIMUM_ROW_GROUP_ROWS = 1_000_000
MATCHUP_SCHEMA = pyarrow.schema(
    [
        ("variable", pyarrow.string()),
        ("depth_bin", pyarrow.string()),
        ("lead_day", pyarrow.int16()),
        ("start_date", pyarrow.string()),
        ("latitude", pyarrow.float32()),
        ("longitude", pyarrow.float32()),
        ("observation_value", pyarrow.float32()),
        ("model_value", pyarrow.float32()),
    ]
)
_MATCHUP_GROUP_COLUMNS = ["start_date", "lead_day", "variable", "depth_bin"]
_MATCHUP_PAIR_COLUMNS = ["start_date", "lead_day"]

_EDDY_CENSUS_SCHEMA_VERSION = "4"
# Contours are simplified (Douglas-Peucker, closed ring) with this tolerance in degrees of latitude
# (about 1.1 km, a 25th of the quarter-degree detection grid), on a local equirectangular projection,
# then the tolerance is doubled until at most _MAXIMUM_CONTOUR_POINT_COUNT vertices are left.
CONTOUR_SIMPLIFICATION_TOLERANCE_DEGREES = 0.01
_MAXIMUM_CONTOUR_POINT_COUNT = 32
_COORDINATE_ROUNDING_DECIMALS = 3
_COORDINATE_UNITS_PER_DEGREE = 10**_COORDINATE_ROUNDING_DECIMALS
EDDY_POLARITIES = ("cyclone", "anticyclone")
_AMPLITUDE_ROUNDING_DECIMALS = 4
_RADIUS_ROUNDING_DECIMALS = 1

_logger = logging.getLogger(__name__)

# Year-mode super-observation grids, global at 2 degrees and IBI at a quarter degree. The block size is
# the edge of the latitude-longitude boxes resampled by the per-start bootstrap: wide enough to hold
# a correlated stretch of altimetry track or a cluster of profiles.
YEAR_GRIDS = {
    "global": {"lat0": -90, "dlat": 2, "nlat": 90, "lon0": -180, "dlon": 2, "nlon": 180, "block_size_degrees": 5},
    "ibi": {
        "lat0": 26.0,
        "dlat": 0.25,
        "nlat": 121,
        "lon0": -19.25,
        "dlon": 0.25,
        "nlon": 98,
        "block_size_degrees": 2,
    },
}
_YEAR_TARGETS = [
    (Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(), "surface", "SSH"),
    (Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(), "surface", "T"),
    (Variable.SEA_WATER_SALINITY.key(), "0-5m", "S"),
    (Variable.EASTWARD_SEA_WATER_VELOCITY.key(), "15m", "u"),
    (Variable.NORTHWARD_SEA_WATER_VELOCITY.key(), "15m", "v"),
]
_YEAR_GEOGRAPHY_DECIMALS = {"SSH": 4, "T": 3, "S": 4, "u": 4, "v": 4}

# Match-ups of one start are correlated along a track and down a profile, so the per-start intervals
# resample latitude-longitude blocks rather than match-ups, and a per-cell bias standard error treats
# each forecast start as one independent value.
BOOTSTRAP_SEED = 20240703
_YEAR_CI_BOOTSTRAP_DRAWS = 1_000
_MINIMUM_BOOTSTRAP_BLOCKS = 10
_MINIMUM_CELL_STARTS = 3
_CONFIDENCE = 0.95


@dataclass(frozen=True)
class ViewerArtifacts:
    insights: dict[str, str | None]
    has_maps: bool


def confidence_interval(bootstrap_values: numpy.ndarray, confidence: float) -> tuple[float, float]:
    finite = bootstrap_values[numpy.isfinite(bootstrap_values)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    low, high = numpy.percentile(finite, [(1.0 - confidence) / 2.0 * 100.0, (1.0 + confidence) / 2.0 * 100.0])
    return (float(low), float(high))


def class4_rmsd_rows(matchups: pandas.DataFrame) -> pandas.DataFrame:
    """Class IV RMSD, count and missing count per variable, depth bin and lead day (0-based)."""
    return pandas.concat(
        [compute_class4_rmsd_table(group, variable) for variable, group in matchups.groupby("variable", sort=False)],
        ignore_index=True,
    )


def class4_rmsd_rows_by_start(matchups: pandas.DataFrame | Class4Matchups) -> pandas.DataFrame:
    parts = class4_matchups(matchups)
    return pandas.concat(
        [class4_rmsd_rows(parts.start(first_day)).assign(first_day=first_day) for first_day in parts.first_days()],
        ignore_index=True,
    )


def _scored(matchups: pandas.DataFrame) -> pandas.DataFrame:
    scored = matchups.dropna(subset=["observation_value", "model_value"])
    return scored.assign(signed_error=scored["model_value"] - scored["observation_value"])


def _json_float(value) -> float | None:
    return None if value is None or numpy.isnan(value) else float(value)


def _serving_matchups(matchups: pandas.DataFrame) -> pyarrow.Table:
    table = pyarrow.table(
        {
            "variable": matchups["variable"].astype(str).to_numpy(dtype=object),
            "depth_bin": matchups["depth_bin"].astype(str).to_numpy(dtype=object),
            "lead_day": matchups["lead_day"].to_numpy() + 1,
            "start_date": pandas.to_datetime(matchups["first_day"]).dt.strftime("%Y-%m-%d").to_numpy(dtype=object),
            "latitude": matchups[Dimension.LATITUDE.key()].to_numpy(),
            "longitude": matchups[Dimension.LONGITUDE.key()].to_numpy(),
            "observation_value": matchups["observation_value"].to_numpy(),
            "model_value": matchups["model_value"].to_numpy(),
        }
    ).cast(MATCHUP_SCHEMA)
    order = pyarrow.compute.sort_indices(table, sort_keys=[(column, "ascending") for column in _MATCHUP_GROUP_COLUMNS])
    return table.take(order)


def _group_boundaries(columns: list[numpy.ndarray]) -> numpy.ndarray:
    changed = numpy.zeros(len(columns[0]), dtype=bool)
    changed[0] = True
    for column in columns:
        changed[1:] |= column[1:] != column[:-1]
    return numpy.flatnonzero(changed)


def _packed_row_group_spans(table: pyarrow.Table) -> list[tuple[int, int]]:
    """``(offset, length)`` of each row group: whole (variable, depth_bin) blocks, packed per pair."""
    row_count = table.num_rows
    if not row_count:
        return []
    block_starts = _group_boundaries(
        [table.column(column).to_numpy(zero_copy_only=False) for column in _MATCHUP_GROUP_COLUMNS]
    )
    pair_starts = set(
        _group_boundaries(
            [table.column(column).to_numpy(zero_copy_only=False) for column in _MATCHUP_PAIR_COLUMNS]
        ).tolist()
    )
    block_ends = numpy.append(block_starts[1:], row_count)
    spans: list[tuple[int, int]] = []
    open_start = None
    for block_start, block_end in zip(block_starts.tolist(), block_ends.tolist()):
        if open_start is not None and block_start in pair_starts:
            spans.append((open_start, block_start - open_start))
            open_start = None
        if open_start is None:
            open_start = block_start
        while block_end - open_start > MAXIMUM_ROW_GROUP_ROWS:
            spans.append((open_start, MAXIMUM_ROW_GROUP_ROWS))
            open_start += MAXIMUM_ROW_GROUP_ROWS
        if block_end - open_start >= TARGET_ROW_GROUP_ROWS:
            spans.append((open_start, block_end - open_start))
            open_start = None
    if open_start is not None:
        spans.append((open_start, row_count - open_start))
    return spans


def write_matchup_parquet(matchups: pandas.DataFrame | Class4Matchups, output_path: Path) -> None:
    """Write the match-ups in the viewer serving layout, one forecast start at a time."""
    parts = class4_matchups(matchups)
    with pyarrow.parquet.ParquetWriter(
        output_path,
        MATCHUP_SCHEMA,
        compression="zstd",
        compression_level=3,
        write_statistics=_MATCHUP_GROUP_COLUMNS,
    ) as writer:
        for first_day in parts.first_days():
            table = _serving_matchups(parts.start(first_day))
            for offset, length in _packed_row_group_spans(table):
                writer.write_table(table.slice(offset, length))
    verify_matchup_parquet(output_path)


def verify_matchup_parquet(output_path: Path) -> None:
    """Check that every row group holds one (start_date, lead_day) pair, in order, with statistics."""
    metadata = pyarrow.parquet.ParquetFile(output_path).metadata
    group_indices = [MATCHUP_SCHEMA.names.index(column) for column in _MATCHUP_GROUP_COLUMNS]
    previous_key = None
    for group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(group_index)
        statistics = [row_group.column(column_index).statistics for column_index in group_indices]
        if any(column_statistics is None for column_statistics in statistics):
            raise ValueError(f"Match-up row group {group_index} has no statistics.")
        if any(column_statistics.min != column_statistics.max for column_statistics in statistics[:2]):
            raise ValueError(f"Match-up row group {group_index} mixes several (start_date, lead_day) pairs.")
        if row_group.num_rows > MAXIMUM_ROW_GROUP_ROWS:
            raise ValueError(f"Match-up row group {group_index} has {row_group.num_rows} rows.")
        key = tuple(column_statistics.min for column_statistics in statistics)
        if previous_key is not None and key < previous_key:
            raise ValueError(f"Match-up row group {group_index} is out of order.")
        previous_key = key


def _lead_day_parts(parts: Class4Matchups):
    for variable in parts.variables():
        for lead_day in parts.lead_days(variable):
            yield parts.part(variable, lead_day)


def rmsd_by_depth(matchups: pandas.DataFrame | Class4Matchups) -> dict:
    """Pooled RMSD, bias and scored count per depth bin and lead day, for variables with several depth bins."""
    parts = class4_matchups(matchups)
    # A group of a variable, depth bin and lead day lies in one part, with its rows in table order.
    part_rows, part_biases = [], []
    for part in _lead_day_parts(parts):
        if len(part):
            part_rows.append(class4_rmsd_rows(part))
            part_biases.append(_scored(part).groupby(["variable", "depth_bin", "lead_day"])["signed_error"].mean())
    rows = pandas.concat(part_rows, ignore_index=True).set_index(["variable", "depth_bin", "lead_day"])
    bias = pandas.concat(part_biases)
    variables = {}
    for variable in rows.index.get_level_values("variable").unique():
        variable_rows = rows.loc[variable]
        depth_bins = sorted(variable_rows.index.get_level_values("depth_bin").unique(), key=DEPTH_BIN_DISPLAY_ORDER.get)
        if len(depth_bins) < 2:
            continue
        lead_days = sorted(variable_rows.index.get_level_values("lead_day").unique())
        rmsd_matrix, bias_matrix, count_matrix = [], [], []
        for depth_bin in depth_bins:
            rmsd_row, bias_row, count_row = [], [], []
            for lead_day in lead_days:
                key = (depth_bin, lead_day)
                count = (
                    int(variable_rows.at[key, "count"] - variable_rows.at[key, "missing"])
                    if key in variable_rows.index
                    else 0
                )
                rmsd_row.append(_json_float(variable_rows.at[key, "rmsd"]) if count else None)
                bias_row.append(_json_float(bias.get((variable, depth_bin, lead_day))) if count else None)
                count_row.append(count)
            rmsd_matrix.append(rmsd_row)
            bias_matrix.append(bias_row)
            count_matrix.append(count_row)
        variables[str(variable)] = {
            "depth_bins": [str(depth_bin) for depth_bin in depth_bins],
            "leads": [int(lead_day) + 1 for lead_day in lead_days],
            "rmsd": rmsd_matrix,
            "bias": bias_matrix,
            "n": count_matrix,
        }
    return variables


def _grid_cells(latitude: numpy.ndarray, longitude: numpy.ndarray, grid: dict) -> tuple[numpy.ndarray, numpy.ndarray]:
    longitude_span = grid["nlon"] * grid["dlon"]
    wrapped_longitude = ((longitude - grid["lon0"]) % longitude_span) + grid["lon0"]
    latitude_bin = numpy.floor((latitude - grid["lat0"]) / grid["dlat"]).astype(numpy.int64)
    longitude_bin = numpy.floor((wrapped_longitude - grid["lon0"]) / grid["dlon"]).astype(numpy.int64)
    valid = (latitude_bin >= 0) & (latitude_bin < grid["nlat"]) & (longitude_bin >= 0) & (longitude_bin < grid["nlon"])
    return latitude_bin * grid["nlon"] + longitude_bin, valid


def _per_start_bias_standard_error(
    cell: numpy.ndarray, start: numpy.ndarray, signed_error: numpy.ndarray, cell_count: int, start_count: int
) -> tuple[numpy.ndarray, numpy.ndarray]:
    cell_start = cell * start_count + start
    count = numpy.bincount(cell_start, minlength=cell_count * start_count).reshape(cell_count, start_count)
    signed_sum = numpy.bincount(cell_start, signed_error, cell_count * start_count).reshape(cell_count, start_count)
    has_data = count > 0
    start_mean = numpy.where(has_data, signed_sum / numpy.maximum(count, 1), 0.0)
    starts_with_data = has_data.sum(axis=1)
    mean_of_means = start_mean.sum(axis=1) / numpy.maximum(starts_with_data, 1)
    squared_deviation = numpy.where(has_data, (start_mean - mean_of_means[:, None]) ** 2, 0.0).sum(axis=1)
    variance = squared_deviation / numpy.maximum(starts_with_data - 1, 1)
    standard_error = numpy.sqrt(variance / numpy.maximum(starts_with_data, 1))
    return numpy.where(starts_with_data >= _MINIMUM_CELL_STARTS, standard_error, numpy.nan), starts_with_data


def _scored_target(parts: Class4Matchups, variable: str, depth_bin: str, lead_day: int) -> pandas.DataFrame:
    scored = _scored(parts.part(variable, lead_day))
    return scored[scored["depth_bin"] == depth_bin]


def year_error_geography(matchups: pandas.DataFrame | Class4Matchups, grid: dict, lead_days_count: int) -> dict:
    """Mean |error|, bias, count, start count and bias standard error per grid cell, over the whole year."""
    cell_count = grid["nlat"] * grid["nlon"]
    parts = class4_matchups(matchups)
    start_dates = numpy.array(parts.scored_first_days(), dtype="datetime64[ns]")
    geography = {
        "grid": grid,
        "variables": {},
        "meta": {
            "n_starts": len(parts.first_days()),
            "generated_from": MATCHUP_PARQUET_FILENAME,
            "depth_bin": {short: depth_bin for _, depth_bin, short in _YEAR_TARGETS},
            "aggregation": "time-mean of |obs-model| per cell",
            "bias_standard_error": (
                "per cell and lead day, the forecast start is the independent unit: bias_se is "
                "std(per-start mean of model-obs, ddof=1)/sqrt(n_starts), with n_starts the number of starts "
                f"with match-ups in the cell; null below {_MINIMUM_CELL_STARTS} starts. bias is the pooled "
                "mean over the n match-ups (shared by the |error| and bias rasters), which weights starts by "
                "their match-up count"
            ),
        },
    }
    for variable, depth_bin, short in _YEAR_TARGETS:
        decimals = _YEAR_GEOGRAPHY_DECIMALS[short]
        leads, bias_leads, count_leads, start_count_leads, bias_se_leads = {}, {}, {}, {}, {}
        for lead_day in range(lead_days_count):
            lead_target = _scored_target(parts, variable, depth_bin, lead_day)
            cell, valid = _grid_cells(
                lead_target[Dimension.LATITUDE.key()].to_numpy(),
                lead_target[Dimension.LONGITUDE.key()].to_numpy(),
                grid,
            )
            signed_error = lead_target["signed_error"].to_numpy()[valid]
            start = numpy.searchsorted(start_dates, lead_target["first_day"].to_numpy(dtype="datetime64[ns]"))[valid]
            cell = cell[valid]
            count = numpy.bincount(cell, minlength=cell_count)
            absolute_sum = numpy.bincount(cell, numpy.abs(signed_error), cell_count)
            signed_sum = numpy.bincount(cell, signed_error, cell_count)
            bias_standard_error, starts_with_data = _per_start_bias_standard_error(
                cell, start, signed_error, cell_count, len(start_dates)
            )
            key = str(lead_day + 1)
            leads[key] = [
                None if cell_n == 0 else round(float(absolute_sum[index] / cell_n), decimals)
                for index, cell_n in enumerate(count)
            ]
            bias_leads[key] = [
                None if cell_n == 0 else round(float(signed_sum[index] / cell_n), decimals)
                for index, cell_n in enumerate(count)
            ]
            count_leads[key] = [int(cell_n) for cell_n in count]
            start_count_leads[key] = [int(cell_starts) for cell_starts in starts_with_data]
            bias_se_leads[key] = [
                None if numpy.isnan(value) else round(float(value), decimals) for value in bias_standard_error
            ]
        geography["variables"][short] = {
            "leads": leads,
            "bias": bias_leads,
            "n": count_leads,
            "n_starts": start_count_leads,
            "bias_se": bias_se_leads,
        }
    return geography


def _bootstrap_blocks(latitude: numpy.ndarray, longitude: numpy.ndarray, block_size_degrees: float) -> numpy.ndarray:
    longitude_block_count = int(numpy.ceil(360.0 / block_size_degrees))
    latitude_block = numpy.floor((latitude + 90.0) / block_size_degrees).astype(numpy.int64)
    longitude_block = numpy.floor(((longitude + 180.0) % 360.0) / block_size_degrees).astype(numpy.int64)
    return latitude_block * longitude_block_count + longitude_block


def _block_bootstrap_intervals(
    block: numpy.ndarray, signed_error: numpy.ndarray, generator: numpy.random.Generator
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    _, block_index = numpy.unique(block, return_inverse=True)
    block_count = int(block_index.max()) + 1 if block_index.size else 0
    if block_count < _MINIMUM_BOOTSTRAP_BLOCKS:
        return None, None
    count = numpy.bincount(block_index, minlength=block_count).astype(float)
    signed_sum = numpy.bincount(block_index, signed_error, block_count)
    square_sum = numpy.bincount(block_index, signed_error * signed_error, block_count)
    draws = generator.multinomial(
        block_count, numpy.full(block_count, 1.0 / block_count), size=_YEAR_CI_BOOTSTRAP_DRAWS
    )
    drawn_count = draws @ count
    rmsd_interval = confidence_interval(numpy.sqrt(draws @ square_sum / drawn_count), _CONFIDENCE)
    bias_interval = confidence_interval(draws @ signed_sum / drawn_count, _CONFIDENCE)
    return rmsd_interval, bias_interval


def _rounded_interval(interval: tuple[float, float] | None) -> tuple[float | None, float | None]:
    if interval is None or not all(numpy.isfinite(interval)):
        return (None, None)
    return (round(interval[0], 6), round(interval[1], 6))


def _bootstrap_generator(variable: str, depth_bin: str, lead_day: int, first_day) -> numpy.random.Generator:
    key = f"{variable}|{depth_bin}|{lead_day}|{pandas.Timestamp(first_day).date().isoformat()}"
    digest = numpy.frombuffer(hashlib.sha256(key.encode()).digest(), dtype="<u4")
    return numpy.random.default_rng(numpy.random.SeedSequence([BOOTSTRAP_SEED, *digest.tolist()]))


def _blocks_and_errors_by_start(
    parts: Class4Matchups, variable: str, depth_bin: str, lead_day: int, block_size_degrees: float
) -> dict:
    target = _scored_target(parts, variable, depth_bin, lead_day)
    blocks = _bootstrap_blocks(
        target[Dimension.LATITUDE.key()].to_numpy(dtype=float),
        target[Dimension.LONGITUDE.key()].to_numpy(dtype=float),
        block_size_degrees,
    )
    return {
        pandas.Timestamp(first_day): (blocks[indices], target["signed_error"].to_numpy()[indices])
        for first_day, indices in target.reset_index(drop=True).groupby("first_day", sort=False).indices.items()
    }


def year_rmsd_by_start(
    matchups: pandas.DataFrame | Class4Matchups, rows_by_start: pandas.DataFrame, grid: dict, lead_days_count: int
) -> dict:
    """Pooled RMSD, bias and their block-bootstrap intervals per start date, for the year chart."""
    parts = class4_matchups(matchups)
    rows = rows_by_start.set_index(["variable", "depth_bin", "lead_day", "first_day"])
    start_dates = parts.scored_first_days()
    payload = {
        "variables": {},
        "meta": {
            "method": "pooled RMSD over all class-4 match-ups per start date (same reduction as official scores)",
            "bias_method": "pooled mean(model - obs) per start date",
            "rmsd_centred_method": (
                "pooled RMSD with the pooled bias of the start date and lead removed: "
                "sqrt(mean squared error - bias^2); for SSH this discards the uniform datum offset"
            ),
            "grid": grid,
            "generated_from": MATCHUP_PARQUET_FILENAME,
            "ci_method": (
                f"rmsd_ci and bias_ci: spatial cluster (block) bootstrap per start date. Match-ups are grouped in "
                f"{grid['block_size_degrees']} degree latitude-longitude blocks, which keeps the correlation "
                f"along a track or down a profile inside a block; each of {_YEAR_CI_BOOTSTRAP_DRAWS} draws "
                f"(seed {BOOTSTRAP_SEED} combined with a SHA-256 hash of the variable, depth bin, lead day and "
                f"start date, so an interval does not change when other start dates are added) takes as many "
                f"blocks as are occupied, with replacement, and pools them: "
                f"RMSD = sqrt(sum of squared errors / count), bias = sum of errors / count. "
                f"{int(_CONFIDENCE * 100)}% percentile intervals; null with fewer than "
                f"{_MINIMUM_BOOTSTRAP_BLOCKS} occupied blocks. Blocks are assumed independent of each other, "
                f"so correlation at scales larger than a block still narrows the interval"
            ),
        },
    }
    for variable, depth_bin, short in _YEAR_TARGETS:
        leads = {}
        for lead_day in range(lead_days_count):
            groups = _blocks_and_errors_by_start(parts, variable, depth_bin, lead_day, grid["block_size_degrees"])
            series = {
                name: []
                for name in [
                    "dates",
                    "rmsd",
                    "rmsd_centred",
                    "n",
                    "bias",
                    "rmsd_ci_low",
                    "rmsd_ci_high",
                    "bias_ci_low",
                    "bias_ci_high",
                ]
            }
            for first_day in start_dates:
                group = groups.get(first_day)
                if group is None:
                    continue
                block, signed_error = group
                rmsd_interval, bias_interval = _block_bootstrap_intervals(
                    block, signed_error, _bootstrap_generator(variable, depth_bin, lead_day, first_day)
                )
                rmsd_low, rmsd_high = _rounded_interval(rmsd_interval)
                bias_low, bias_high = _rounded_interval(bias_interval)
                series["dates"].append(pandas.Timestamp(first_day).date().isoformat())
                series["rmsd"].append(round(float(rows.at[(variable, depth_bin, lead_day, first_day), "rmsd"]), 6))
                series["rmsd_centred"].append(round(float(signed_error.std()), 6))
                series["n"].append(int(signed_error.size))
                series["bias"].append(round(float(signed_error.mean()), 6))
                series["rmsd_ci_low"].append(rmsd_low)
                series["rmsd_ci_high"].append(rmsd_high)
                series["bias_ci_low"].append(bias_low)
                series["bias_ci_high"].append(bias_high)
            leads[str(lead_day + 1)] = series
        payload["variables"][short] = {"depth_bin": depth_bin, "leads": leads}
    return payload


def _douglas_peucker_keep(points: numpy.ndarray, tolerance: float) -> numpy.ndarray:
    """Mask of the vertices of an open polyline kept by Douglas-Peucker; both ends are kept."""
    keep = numpy.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    pending = [(0, len(points) - 1)]
    while pending:
        first, last = pending.pop()
        if last <= first + 1:
            continue
        start, chord = points[first], points[last] - points[first]
        inner = points[first + 1 : last] - start
        length = float(numpy.hypot(chord[0], chord[1]))
        if length == 0.0:
            distances = numpy.hypot(inner[:, 0], inner[:, 1])
        else:
            distances = numpy.abs(chord[0] * inner[:, 1] - chord[1] * inner[:, 0]) / length
        farthest = int(numpy.argmax(distances))
        if distances[farthest] > tolerance:
            middle = first + 1 + farthest
            keep[middle] = True
            pending.extend([(first, middle), (middle, last)])
    return keep


def simplified_ring(
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
    tolerance_degrees: float = CONTOUR_SIMPLIFICATION_TOLERANCE_DEGREES,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Vertices of a closed contour simplified by Douglas-Peucker, without the closing duplicate.

    Distances are measured in degrees of latitude on a local equirectangular projection (longitudes
    scaled by the cosine of the mean latitude). The ring is split at the vertex farthest from the first
    one and each half simplified; the tolerance is doubled until at most
    ``_MAXIMUM_CONTOUR_POINT_COUNT`` vertices remain.
    """
    if latitudes.size > 1 and latitudes[0] == latitudes[-1] and longitudes[0] == longitudes[-1]:
        latitudes, longitudes = latitudes[:-1], longitudes[:-1]
    if latitudes.size <= 3:
        return latitudes, longitudes
    cosine = max(numpy.cos(numpy.radians(float(numpy.mean(latitudes)))), 1e-6)
    points = numpy.column_stack([longitudes * cosine, latitudes])
    split = int(numpy.argmax(numpy.hypot(*(points - points[0]).T)))
    closed = numpy.vstack([points, points[:1]])
    tolerance = tolerance_degrees
    while True:
        keep = numpy.zeros(latitudes.size, dtype=bool)
        keep[: split + 1] |= _douglas_peucker_keep(closed[: split + 1], tolerance)
        keep[split:] |= _douglas_peucker_keep(closed[split:], tolerance)[:-1]
        if keep.sum() <= _MAXIMUM_CONTOUR_POINT_COUNT:
            return latitudes[keep], longitudes[keep]
        tolerance *= 2.0


def _clamp(value, low: float, high: float):
    if isinstance(value, list):
        return [min(high, max(low, element)) for element in value]
    return min(high, max(low, value))


def _eddy(eddy_index: int, eddy_row: pandas.Series) -> dict:
    """
    One eddy with its centre longitude wrapped into [-180, 180) and its contour simplified.

    Amplitude and radii are py-eddy-tracker's values for the full contour. The contour stays
    continuous around the centre and may overhang the dateline (e.g. 180.4); the reader draws it
    on both sides.
    """
    latitudes, longitudes = simplified_ring(
        numpy.asarray(eddy_row[eddy_census.CONTOUR_LATITUDES_COLUMN], dtype=float),
        numpy.asarray(eddy_row[eddy_census.CONTOUR_LONGITUDES_COLUMN], dtype=float),
    )
    native_longitude = round(float(eddy_row[eddy_census.LONGITUDE_COLUMN]), _COORDINATE_ROUNDING_DECIMALS)
    longitude = round((native_longitude + 180.0) % 360.0 - 180.0, _COORDINATE_ROUNDING_DECIMALS)
    longitude_shift = longitude - native_longitude
    eddy = {
        "id": int(eddy_index),
        "latitude": _clamp(
            round(float(eddy_row[eddy_census.LATITUDE_COLUMN]), _COORDINATE_ROUNDING_DECIMALS), -90.0, 90.0
        ),
        "longitude": longitude,
        "polarity": str(eddy_row[eddy_census.POLARITY_COLUMN]),
        "amplitude": round(float(eddy_row[eddy_census.AMPLITUDE_COLUMN]), _AMPLITUDE_ROUNDING_DECIMALS),
        "effective_radius_km": round(float(eddy_row[eddy_census.EFFECTIVE_RADIUS_COLUMN]), _RADIUS_ROUNDING_DECIMALS),
        "speed_radius_km": round(float(eddy_row[eddy_census.SPEED_RADIUS_COLUMN]), _RADIUS_ROUNDING_DECIMALS),
        "contour_latitude": _clamp(
            [round(float(value), _COORDINATE_ROUNDING_DECIMALS) for value in latitudes], -90.0, 90.0
        ),
        "contour_longitude": [
            round(float(value) + longitude_shift, _COORDINATE_ROUNDING_DECIMALS) for value in longitudes
        ],
    }
    return eddy


def _encoded_integers(values: list[int]) -> str:
    """Integers in the encoded polyline format (zig-zag sign, 5-bit groups, offset 63)."""
    characters = []
    for value in values:
        remainder = ~(value << 1) if value < 0 else value << 1
        while remainder >= 0x20:
            characters.append(chr((0x20 | (remainder & 0x1F)) + 63))
            remainder >>= 5
        characters.append(chr(remainder + 63))
    return "".join(characters)


def _decoded_integers(text: str) -> list[int]:
    values, accumulated, shift = [], 0, 0
    for character in text:
        chunk = ord(character) - 63
        accumulated |= (chunk & 0x1F) << shift
        shift += 5
        if chunk < 0x20:
            values.append(~(accumulated >> 1) if accumulated & 1 else accumulated >> 1)
            accumulated, shift = 0, 0
    return values


def _units(value: float) -> int:
    return int(round(value * _COORDINATE_UNITS_PER_DEGREE))


def _encoded_contour(eddy: dict) -> str:
    """Contour vertices as successive (latitude, longitude) steps in millidegrees, the first from the centre."""
    previous_latitude, previous_longitude = _units(eddy["latitude"]), _units(eddy["longitude"])
    steps = []
    for latitude, longitude in zip(eddy["contour_latitude"], eddy["contour_longitude"]):
        latitude_units, longitude_units = _units(latitude), _units(longitude)
        steps += [latitude_units - previous_latitude, longitude_units - previous_longitude]
        previous_latitude, previous_longitude = latitude_units, longitude_units
    return _encoded_integers(steps)


def encoded_eddy_frame(frame: dict) -> dict:
    """
    Columnar census frame (schema 4) of a frame of eddy dicts: one array per field, polarity as an
    index into ``polarities`` and each contour as an encoded polyline string (see ``decoded_eddy_frame``).
    """
    detections = frame["detections"]
    encoded = {
        "lead_day": frame["lead_day"],
        "count": len(detections),
        "polarities": list(EDDY_POLARITIES),
        "latitude": [eddy["latitude"] for eddy in detections],
        "longitude": [eddy["longitude"] for eddy in detections],
        "polarity": [EDDY_POLARITIES.index(eddy["polarity"]) for eddy in detections],
        "amplitude": [eddy["amplitude"] for eddy in detections],
        "effective_radius_km": [eddy["effective_radius_km"] for eddy in detections],
        "speed_radius_km": [eddy["speed_radius_km"] for eddy in detections],
        "contour": [_encoded_contour(eddy) for eddy in detections],
    }
    validate_against_schema(encoded, "eddy")
    return encoded


def decoded_eddy_frame(encoded: dict) -> dict:
    """The frame of eddy dicts (id, centre, polarity, amplitude, radii, contour) of a schema 4 frame."""
    detections = []
    for index in range(encoded["count"]):
        latitude_units = _units(encoded["latitude"][index])
        longitude_units = _units(encoded["longitude"][index])
        steps = _decoded_integers(encoded["contour"][index])
        contour_latitude, contour_longitude = [], []
        for latitude_step, longitude_step in zip(steps[0::2], steps[1::2]):
            latitude_units += latitude_step
            longitude_units += longitude_step
            contour_latitude.append(latitude_units / _COORDINATE_UNITS_PER_DEGREE)
            contour_longitude.append(longitude_units / _COORDINATE_UNITS_PER_DEGREE)
        detections.append(
            {
                "id": index,
                "latitude": encoded["latitude"][index],
                "longitude": encoded["longitude"][index],
                "polarity": encoded["polarities"][encoded["polarity"][index]],
                "amplitude": encoded["amplitude"][index],
                "effective_radius_km": encoded["effective_radius_km"][index],
                "speed_radius_km": encoded["speed_radius_km"][index],
                "contour_latitude": contour_latitude,
                "contour_longitude": contour_longitude,
            }
        )
    return {"lead_day": encoded["lead_day"], "detections": detections}


def _centres_inside(eddies: pandas.DataFrame, region: RegionSpec) -> numpy.ndarray:
    bounds = region.bounds
    latitudes = eddies[eddy_census.LATITUDE_COLUMN].to_numpy(dtype=float)
    longitudes = (eddies[eddy_census.LONGITUDE_COLUMN].to_numpy(dtype=float) + 180.0) % 360.0 - 180.0
    return (
        (latitudes >= bounds.minimum_latitude)
        & (latitudes <= bounds.maximum_latitude)
        & (longitudes >= bounds.minimum_longitude)
        & (longitudes <= bounds.maximum_longitude)
    )


def _eddy_frame(eddies: pandas.DataFrame, lead_day_index: int, centre_region: RegionSpec | None) -> dict:
    kept = eddies if centre_region is None else eddies.loc[_centres_inside(eddies, centre_region)]
    return {
        "lead_day": lead_day_index + 1,
        "detections": [_eddy(eddy_index, eddy_row) for eddy_index, (_, eddy_row) in enumerate(kept.iterrows())],
    }


def eddy_census_frames(dataset: xarray.Dataset, start_index: int, centre_region: RegionSpec | None) -> list[dict]:
    """
    One frame per lead day with the eddies py-eddy-tracker detects in one forecast start.

    ``centre_region`` keeps only the eddies centred in a region when ``dataset`` is a crop padded around it.
    """
    return [
        _eddy_frame(eddy_census.detect_eddies(dataset, start_index, lead_day_index), lead_day_index, centre_region)
        for lead_day_index in range(dataset.sizes[Dimension.LEAD_DAY_INDEX.key()])
    ]


def _all_eddy_frames(dataset: xarray.Dataset, centre_region: RegionSpec | None, max_workers: int | None):
    """``(start_index, frame)`` of every start and lead day, in order, the fields detected in parallel."""
    keys: list[tuple[int, int]] = []

    def fields():
        for start_index, lead_day_index, field, date in eddy_census.challenger_fields(dataset):
            keys.append((start_index, lead_day_index))
            yield field, date

    for position, eddies in enumerate(eddy_census.detect_eddies_in_fields(fields(), max_workers)):
        start_index, lead_day_index = keys[position]
        yield start_index, _eddy_frame(eddies, lead_day_index, centre_region)


def write_eddy_census(
    dataset: xarray.Dataset,
    output_path: Path,
    dataset_slug: str,
    centre_region: RegionSpec | None,
    max_workers: int | None = None,
) -> None:
    """
    Write one ``eddies-start-<date>-lead-<N>.json`` per start and lead day, and the index at ``output_path``.

    The fields are detected in ``max_workers`` processes (``eddy_census.census_max_workers()`` by default);
    the files are the same whatever the number of workers.
    """
    metadata = {
        "kind": "eddy-census",
        "schema_version": _EDDY_CENSUS_SCHEMA_VERSION,
        "variable": Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
        "dataset": dataset_slug,
        "method": eddy_census.PY_EDDY_TRACKER_NAME,
        "parameters": {
            **eddy_census.detection_parameters(),
            "py_eddy_tracker_version": eddy_census.py_eddy_tracker_version(),
            "oceanbench_version": __version__,
        },
        "detection_grid": eddy_census.detection_grid_description(dataset),
    }
    start_dates = [
        pandas.Timestamp(first_day).date().isoformat()
        for first_day in dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    ]
    lead_entries_by_start: list[list[dict]] = [[] for _ in start_dates]
    for start_index, frame in _all_eddy_frames(dataset, centre_region, max_workers):
        start_date = start_dates[start_index]
        file_name = f"eddies-start-{start_date}-lead-{frame['lead_day']}.json"
        payload = {**metadata, "start_index": start_index, "start_date": start_date, "frame": encoded_eddy_frame(frame)}
        (output_path.parent / file_name).write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        lead_entries_by_start[start_index].append({"lead_day": frame["lead_day"], "file": file_name})
    start_entries = [
        {"start_index": start_index, "start_date": start_date, "leads": lead_entries_by_start[start_index]}
        for start_index, start_date in enumerate(start_dates)
    ]
    # "leads" repeats the first start for readers of the previous index shape.
    index = {**metadata, "starts": start_entries, "leads": start_entries[0]["leads"] if start_entries else []}
    output_path.write_text(json.dumps(index, sort_keys=True, indent=2), encoding="utf-8")


def _eddy_census_is_possible(challenger_dataset: xarray.Dataset) -> bool:
    if not eddy_census.has_census_resolution(challenger_dataset):
        return False
    if not eddy_census.py_eddy_tracker_is_installed():
        # Progress goes to the buffered stdout and warnings to stderr: flush first so that, in a log
        # combining both, the warning does not land in the middle of a line still in the buffer.
        sys.stdout.flush()
        _logger.warning("Skipping the eddy census. %s", eddy_census.PY_EDDY_TRACKER_INSTALL_HINT)
        return False
    return True


def write_region_eddy_census(
    insights_directory: Path, dataset_slug: str, challenger_dataset: xarray.Dataset, region: RegionSpec
) -> str | None:
    """
    Write the eddy census of one challenger and region into its insights directory.

    Returns the index file name, or ``None`` when no census is possible (grid too coarse or
    py-eddy-tracker not installed).
    """
    if not _eddy_census_is_possible(challenger_dataset):
        return None
    # A regional census detects on a padded crop so the filter and contours see past the region edge.
    is_global = region.bounds is None
    eddy_dataset = subset_dataset_to_region(
        challenger_dataset, padded_region(region, eddy_census.REGIONAL_CENSUS_MARGIN_DEGREES)
    )
    write_eddy_census(
        eddy_dataset, insights_directory / EDDY_CENSUS_FILENAME, dataset_slug, None if is_global else region
    )
    return EDDY_CENSUS_FILENAME


def _write_json(document: dict, path: Path) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def _stage_marker(stage: str) -> None:
    # A UTC timestamp to line the viewer stages up with a memory timeline, when OCEANBENCH_STAGE_MARKERS is set.
    if not os.environ.get("OCEANBENCH_STAGE_MARKERS"):
        return
    print(f"[viewer stage] {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ} {stage}", file=sys.stderr, flush=True)


def write_viewer_artifacts(
    directory: Path,
    dataset_slug: str,
    challenger_dataset: xarray.Dataset,
    region: RegionSpec,
    matchups: pandas.DataFrame | Class4Matchups | None,
    rows_by_start: pandas.DataFrame | None,
    map_dataset: xarray.Dataset | None = None,
) -> ViewerArtifacts:
    """
    Write the viewer files of one challenger and region under ``directory``.

    The maps (pyramid and water-column store) are written for the global region only, as
    ``<slug>.zarr``, ``<slug>.viewer-manifest.json`` and ``<slug>.columns.zarr``. The insights are
    written under ``insights/<slug>/<region>/``; the year files need a super-observation grid, so
    they are written for the official regions only.
    """
    _stage_marker(f"viewer artifacts {region.id} start")
    insights_directory = directory / "insights" / dataset_slug / region.id
    insights_directory.mkdir(parents=True, exist_ok=True)
    insights: dict[str, str | None] = {
        "class4_matchups": None,
        "eddies": None,
        "rmsd_by_depth": None,
        "year_error_geography": None,
        "year_rmsd_by_start": None,
    }
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]

    if matchups is not None and len(matchups):
        write_matchup_parquet(matchups, insights_directory / MATCHUP_PARQUET_FILENAME)
        insights["class4_matchups"] = MATCHUP_PARQUET_FILENAME
        variables = rmsd_by_depth(matchups)
        if variables:
            document = {"schema_version": 1, "challenger": dataset_slug, "region": region.id, "variables": variables}
            _write_json(document, insights_directory / RMSD_BY_DEPTH_FILENAME)
            insights["rmsd_by_depth"] = RMSD_BY_DEPTH_FILENAME
        if region.id in YEAR_GRIDS:
            grid = YEAR_GRIDS[region.id]
            _write_json(
                year_error_geography(matchups, grid, lead_days_count),
                insights_directory / YEAR_ERROR_GEOGRAPHY_FILENAME,
            )
            _write_json(
                year_rmsd_by_start(matchups, rows_by_start, grid, lead_days_count),
                insights_directory / YEAR_RMSD_BY_START_FILENAME,
            )
            insights["year_error_geography"] = YEAR_ERROR_GEOGRAPHY_FILENAME
            insights["year_rmsd_by_start"] = YEAR_RMSD_BY_START_FILENAME

    _stage_marker(f"eddy census {region.id} start")
    insights["eddies"] = write_region_eddy_census(insights_directory, dataset_slug, challenger_dataset, region)
    _stage_marker(f"eddy census {region.id} end")

    has_maps = region.id == GLOBAL_REGION_NAME
    if has_maps:
        _stage_marker("viewer maps start")
        year = pandas.Timestamp(challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values[0]).year
        # ``map_dataset`` is the same challenger opened in larger chunks, which keeps the map graphs small.
        map_dataset = challenger_dataset if map_dataset is None else map_dataset
        build_column_store(
            map_dataset,
            output_path=str(directory / f"{dataset_slug}{COLUMN_STORE_SUFFIX}"),
            dataset_slug=dataset_slug,
            year=year,
        )
        layers, specs = viewer_layers(map_dataset)
        build_pyramid(
            layers,
            specs,
            output_path=str(directory / f"{dataset_slug}{PYRAMID_SUFFIX}"),
            dataset_slug=dataset_slug,
            year=year,
            has_columns=True,
            kind=get_challenger_kind(challenger_dataset),
            description=get_challenger_description(challenger_dataset),
        )
        _stage_marker("viewer maps end")
    return ViewerArtifacts(
        insights={
            key: None if file_name is None else f"./data/insights/{dataset_slug}/{region.id}/{file_name}"
            for key, file_name in insights.items()
        },
        has_maps=has_maps,
    )
