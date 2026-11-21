# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass

import xarray

DATASET_SOURCE_KIND_ATTRIBUTE = "oceanbench_source_kind"
DATASET_SOURCE_NAME_ATTRIBUTE = "oceanbench_source_name"
DATASET_SOURCE_RESOLUTION_ATTRIBUTE = "oceanbench_source_resolution"
DATASET_SOURCE_VARIANT_ATTRIBUTE = "oceanbench_source_variant"


@dataclass(frozen=True)
class DatasetSource:
    kind: str
    name: str
    resolution: str | None = None
    variant: str | None = None


def with_dataset_source(
    dataset: xarray.Dataset,
    *,
    kind: str,
    name: str,
    resolution: str | None = None,
    variant: str | None = None,
) -> xarray.Dataset:
    attrs = dict(dataset.attrs)
    attrs[DATASET_SOURCE_KIND_ATTRIBUTE] = kind
    attrs[DATASET_SOURCE_NAME_ATTRIBUTE] = name
    if resolution is None:
        attrs.pop(DATASET_SOURCE_RESOLUTION_ATTRIBUTE, None)
    else:
        attrs[DATASET_SOURCE_RESOLUTION_ATTRIBUTE] = resolution
    if variant is None:
        attrs.pop(DATASET_SOURCE_VARIANT_ATTRIBUTE, None)
    else:
        attrs[DATASET_SOURCE_VARIANT_ATTRIBUTE] = variant
    return dataset.assign_attrs(attrs)


def get_dataset_source(dataset: xarray.Dataset) -> DatasetSource | None:
    kind = dataset.attrs.get(DATASET_SOURCE_KIND_ATTRIBUTE)
    name = dataset.attrs.get(DATASET_SOURCE_NAME_ATTRIBUTE)
    if kind in (None, "") or name in (None, ""):
        return None
    resolution = dataset.attrs.get(DATASET_SOURCE_RESOLUTION_ATTRIBUTE)
    variant = dataset.attrs.get(DATASET_SOURCE_VARIANT_ATTRIBUTE)
    return DatasetSource(kind=kind, name=name, resolution=resolution, variant=variant)


CHALLENGER_KIND_ATTRIBUTE = "oceanbench_challenger_kind"
CHALLENGER_DESCRIPTION_ATTRIBUTE = "oceanbench_challenger_description"
CHALLENGER_KINDS = ("model", "baseline", "reference")
DEFAULT_CHALLENGER_KIND = "model"


def with_challenger_kind(dataset: xarray.Dataset, kind: str, description: str) -> xarray.Dataset:
    """Mark a challenger as a forecast ``model``, a ``baseline`` or a ``reference``, with a short description."""
    if kind not in CHALLENGER_KINDS:
        raise ValueError(f"Unknown challenger kind {kind!r}, expected one of {CHALLENGER_KINDS}")
    return dataset.assign_attrs({CHALLENGER_KIND_ATTRIBUTE: kind, CHALLENGER_DESCRIPTION_ATTRIBUTE: description})


def get_challenger_kind(dataset: xarray.Dataset) -> str:
    return dataset.attrs.get(CHALLENGER_KIND_ATTRIBUTE, DEFAULT_CHALLENGER_KIND)


def get_challenger_description(dataset: xarray.Dataset) -> str | None:
    return dataset.attrs.get(CHALLENGER_DESCRIPTION_ATTRIBUTE)


# Challengers built on GLORYS fields carry this datum marker so Class IV SLA uses the GLORYS shift.
SEA_SURFACE_HEIGHT_DATUM_ATTRIBUTE = "oceanbench_sea_surface_height_datum"
GLORYS_SEA_SURFACE_HEIGHT_DATUM = "glorys"


def with_glorys_sea_surface_height_datum(dataset: xarray.Dataset) -> xarray.Dataset:
    return dataset.assign_attrs({SEA_SURFACE_HEIGHT_DATUM_ATTRIBUTE: GLORYS_SEA_SURFACE_HEIGHT_DATUM})
