# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Mixed layer depth and geostrophic currents derived per ensemble member.

A derived quantity is a non-linear function of the fields it is derived from, so deriving it
from the ensemble mean is not the same thing as deriving it from every member. The mean
density profile of a spread-out ensemble is smoother than any member profile and crosses the
mixed layer threshold deeper than the members it came from, and the mean sea surface height is
flatter than any member, so its geostrophic velocities are weaker. Deriving from the ensemble
mean would therefore report a single field that no member holds, and it would carry no spread
at all. Every member is derived on its own instead, and the resulting member dimension is
scored with the ensemble metrics of :mod:`oceanbench.core.ensemble_gridded`, exactly as a raw
variable is.

The deterministic kernels are used untouched, one member slice at a time. Both of them write
their output into a dataset whose dimensions they name explicitly, so an extra leading
dimension does not fit them and the member loop belongs outside: one slice in, one
deterministic derivation out, concatenated back along the member dimension.

The reference derivation is deterministic. The GLORYS fields carry no member dimension, so
they go through the same kernels once, through :func:`reference_mixed_layer_depth` and
:func:`reference_geostrophic_currents`, and the ensemble and the reference are derived the
same way by construction.
"""

from collections.abc import Callable

import xarray

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Variable
from oceanbench.core.derived_quantities import compute_geostrophic_currents, compute_mixed_layer_depth
from oceanbench.core.ensemble_gridded import ENSEMBLE_DIMENSION, EnsembleFieldStatistics, ensemble_field_statistics

MIXED_LAYER_DEPTH_INPUT_VARIABLE_KEYS = (
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
    Variable.SEA_WATER_SALINITY.key(),
)
GEOSTROPHIC_CURRENTS_INPUT_VARIABLE_KEYS = (Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),)

DERIVED_VARIABLE_KEYS = (
    Variable.MIXED_LAYER_DEPTH.key(),
    Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY.key(),
    Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY.key(),
)


def _derive_per_member(
    dataset: xarray.Dataset,
    derive: Callable[[xarray.Dataset], xarray.Dataset],
    ensemble_dimension: str,
    required_variable_keys: tuple[str, ...],
) -> xarray.Dataset:
    held_variable_keys = set(rename_dataset_with_standard_names(dataset).data_vars)
    missing_variable_keys = [key for key in required_variable_keys if key not in held_variable_keys]
    if missing_variable_keys:
        raise ValueError(
            f"the dataset to derive per member does not hold {missing_variable_keys}, "
            f"it holds {sorted(held_variable_keys)}"
        )
    member_count = dataset.sizes[ensemble_dimension]
    derived = xarray.concat(
        [derive(dataset.isel({ensemble_dimension: member_index})) for member_index in range(member_count)],
        dim=ensemble_dimension,
    )
    # Concatenating member by member leaves one chunk per member, and the ensemble metrics
    # reduce over the whole member dimension at once, which a chunked core dimension forbids.
    return derived.chunk({ensemble_dimension: -1})


def per_member_mixed_layer_depth(
    dataset: xarray.Dataset,
    *,
    ensemble_dimension: str = ENSEMBLE_DIMENSION,
) -> xarray.Dataset:
    """Mixed layer depth of every member, keeping the member dimension.

    Each member slice goes through the deterministic kernel of
    :mod:`oceanbench.core.mixed_layer_depth`, so the density threshold and the depth cap are
    the ones the deterministic axis publishes.
    """
    return _derive_per_member(
        dataset,
        compute_mixed_layer_depth,
        ensemble_dimension,
        MIXED_LAYER_DEPTH_INPUT_VARIABLE_KEYS,
    )


def per_member_geostrophic_currents(
    dataset: xarray.Dataset,
    *,
    ensemble_dimension: str = ENSEMBLE_DIMENSION,
) -> xarray.Dataset:
    """Geostrophic velocities of every member, keeping the member dimension.

    Each member slice goes through the deterministic kernel of
    :mod:`oceanbench.core.geostrophic_currents`, so the sea surface height derivation and the
    equator exclusion are the ones the deterministic axis publishes.
    """
    return _derive_per_member(
        dataset,
        compute_geostrophic_currents,
        ensemble_dimension,
        GEOSTROPHIC_CURRENTS_INPUT_VARIABLE_KEYS,
    )


def reference_mixed_layer_depth(dataset: xarray.Dataset) -> xarray.Dataset:
    """Mixed layer depth of the deterministic reference, derived once."""
    return compute_mixed_layer_depth(dataset)


def reference_geostrophic_currents(dataset: xarray.Dataset) -> xarray.Dataset:
    """Geostrophic velocities of the deterministic reference, derived once."""
    return compute_geostrophic_currents(dataset)


def derived_quantity_statistics(
    per_member_derived: xarray.Dataset,
    reference_derived: xarray.Dataset,
    *,
    ensemble_dimension: str = ENSEMBLE_DIMENSION,
) -> dict[str, EnsembleFieldStatistics]:
    """Ensemble gridded statistics of every derived variable held by both datasets.

    The values are the same :class:`~oceanbench.core.ensemble_gridded.EnsembleFieldStatistics`
    the raw variables are reduced to, so a derived quantity is published through
    :func:`~oceanbench.core.ensemble_gridded.ensemble_gridded_records` unchanged.
    """
    common_variable_keys = [
        variable_key
        for variable_key in DERIVED_VARIABLE_KEYS
        if variable_key in per_member_derived and variable_key in reference_derived
    ]
    if not common_variable_keys:
        raise ValueError(
            "no derived quantity is held by both datasets: the members hold "
            f"{sorted(per_member_derived.data_vars)} and the reference holds "
            f"{sorted(reference_derived.data_vars)}"
        )
    return {
        variable_key: ensemble_field_statistics(
            per_member_derived[variable_key],
            reference_derived[variable_key],
            ensemble_dimension=ensemble_dimension,
        )
        for variable_key in common_variable_keys
    }
