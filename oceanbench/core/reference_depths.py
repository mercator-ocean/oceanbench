# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import hashlib

import numpy
import xarray

REFERENCE_DEPTH_GRID_ROUNDING_DECIMALS = 3
REFERENCE_DEPTH_GRID_HASH_ATTRIBUTE = "oceanbench_reference_depth_grid_hash"
REFERENCE_DEPTH_GRID_ROUNDING_ATTRIBUTE = "oceanbench_reference_depth_grid_rounding_decimals"
REFERENCE_TARGET_DEPTHS_ATTRIBUTE = "oceanbench_reference_target_depths_m"


def normalised_reference_depths(target_depths: numpy.ndarray) -> numpy.ndarray:
    depths = numpy.asarray(target_depths, dtype=numpy.float64)
    if depths.ndim != 1:
        raise ValueError("Reference target depths must be a one-dimensional array.")
    if depths.size == 0:
        raise ValueError("Reference target depths cannot be empty.")
    if not numpy.all(numpy.isfinite(depths)):
        raise ValueError("Reference target depths must be finite.")
    return numpy.round(depths, REFERENCE_DEPTH_GRID_ROUNDING_DECIMALS)


def reference_depth_grid_hash(target_depths: numpy.ndarray) -> str:
    rounded_depths = normalised_reference_depths(target_depths)
    payload = ",".join(f"{depth:.{REFERENCE_DEPTH_GRID_ROUNDING_DECIMALS}f}" for depth in rounded_depths)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:12]


def reference_depth_grid_variant(target_depths: numpy.ndarray) -> str:
    rounded_depths = normalised_reference_depths(target_depths)
    return f"depths-{len(rounded_depths)}-{reference_depth_grid_hash(rounded_depths)}"


def with_reference_depth_grid_metadata(dataset: xarray.Dataset, target_depths: numpy.ndarray) -> xarray.Dataset:
    rounded_depths = normalised_reference_depths(target_depths)
    attrs = dict(dataset.attrs)
    attrs[REFERENCE_DEPTH_GRID_HASH_ATTRIBUTE] = reference_depth_grid_hash(rounded_depths)
    attrs[REFERENCE_DEPTH_GRID_ROUNDING_ATTRIBUTE] = REFERENCE_DEPTH_GRID_ROUNDING_DECIMALS
    attrs[REFERENCE_TARGET_DEPTHS_ATTRIBUTE] = rounded_depths.tolist()
    return dataset.assign_attrs(attrs)
