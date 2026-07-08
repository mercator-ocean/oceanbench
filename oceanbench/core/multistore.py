# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Picklable recipe to re-open a multi-store concat dataset in a fresh worker process.

A challenger dataset is a concat of ~52 weekly zarr stores and an observation dataset is a
concat of ~370 daily zarr stores; neither carries a single ``encoding['source']`` xarray can
re-open, so the Class-4 match-up parallelism could never reconstruct them in a spawned worker and
always fell back to the serial path. This module threads the ORIGINAL opener recipe (the ordered
member stores, the concat dimension and coordinate, the per-member opener and any post-concat
transform) from the dataset-opening code to the match-up runner, so a worker rebuilds the same lazy
concat and selects only its own starts. The recipe is attached to the in-memory dataset through a
process-local registry keyed by an opaque token in the dataset attributes, which survives the
``isel``/``sel``/``interp`` subsetting applied downstream.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import importlib

import numpy
import xarray

_RECIPE_TOKEN_ATTRIBUTE = "oceanbench_multistore_recipe_token"
_recipe_registry: dict[str, "MultiStoreConcatRecipe"] = {}
_recipe_token_counter = 0


@dataclass(frozen=True)
class OneDegreeInterpolation:
    """The small target grid of the ``*_1_degree`` interpolation, reapplied identically by a worker."""

    latitude: tuple[float, ...]
    longitude: tuple[float, ...]


@dataclass(frozen=True)
class MultiStoreConcatRecipe:
    """Picklable recipe to reconstruct a multi-store concat dataset from its ORIGINAL member stores.

    ``member_opener`` and ``post_process`` are ``"module:function"`` references to module-level,
    picklable functions; ``member_opener`` opens one member store lazily and ``post_process`` (when
    present) applies the deterministic post-concat preparation (renames, standard-name attributes,
    the observation index coordinate). ``interpolation`` reapplies the one-degree interpolation.
    """

    member_stores: tuple[str, ...]
    member_opener: str
    concat_dimension: str
    concat_coordinate: tuple | None = None
    rename: tuple[tuple[str, str], ...] = ()
    assign_index_dimension: str | None = None
    post_process: str | None = None
    interpolation: OneDegreeInterpolation | None = None
    member_open_arguments: tuple = field(default=())


def _resolve_reference(reference: str) -> Callable:
    module_name, function_name = reference.split(":")
    return getattr(importlib.import_module(module_name), function_name)


def open_zarr_member(store: str) -> xarray.Dataset:
    """Open one already-prepared member store (a staged weekly zarr) lazily (dask-backed)."""
    return xarray.open_zarr(store)


def apply_one_degree_interpolation(
    dataset: xarray.Dataset, interpolation: OneDegreeInterpolation
) -> xarray.Dataset:
    from oceanbench.core.interpolate import apply_one_degree_interpolation as _apply

    return _apply(dataset, numpy.asarray(interpolation.latitude), numpy.asarray(interpolation.longitude))


def open_multistore_dataset(recipe: MultiStoreConcatRecipe) -> xarray.Dataset:
    """Rebuild the lazy multi-store concat dataset from ``recipe`` (used in the parent and in workers)."""
    open_member = _resolve_reference(recipe.member_opener)
    members = [open_member(store, *recipe.member_open_arguments) for store in recipe.member_stores]
    combined = members[0] if len(members) == 1 else xarray.concat(members, dim=recipe.concat_dimension)
    if recipe.concat_coordinate is not None:
        combined = combined.assign_coords(
            {recipe.concat_dimension: numpy.asarray(recipe.concat_coordinate)}
        )
    if recipe.rename:
        combined = combined.rename(dict(recipe.rename))
    if recipe.assign_index_dimension is not None:
        combined = combined.assign_coords(
            {
                recipe.assign_index_dimension: (
                    recipe.assign_index_dimension,
                    numpy.arange(combined.sizes[recipe.assign_index_dimension]),
                )
            }
        )
    if recipe.post_process is not None:
        combined = _resolve_reference(recipe.post_process)(combined)
    if recipe.interpolation is not None:
        combined = apply_one_degree_interpolation(combined, recipe.interpolation)
    return combined


def attach_multistore_recipe(
    dataset: xarray.Dataset, recipe: MultiStoreConcatRecipe
) -> xarray.Dataset:
    """Register ``recipe`` for ``dataset`` and stamp the dataset with the lookup token."""
    global _recipe_token_counter
    _recipe_token_counter += 1
    token = f"multistore-{_recipe_token_counter}"
    _recipe_registry[token] = recipe
    return dataset.assign_attrs({_RECIPE_TOKEN_ATTRIBUTE: token})


def get_multistore_recipe(dataset: xarray.Dataset) -> MultiStoreConcatRecipe | None:
    """Return the recipe registered for ``dataset`` (surviving downstream subsetting), or ``None``."""
    token = dataset.attrs.get(_RECIPE_TOKEN_ATTRIBUTE)
    if not isinstance(token, str):
        return None
    return _recipe_registry.get(token)
