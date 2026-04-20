# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass

import xarray

DATASET_SOURCE_KIND_ATTRIBUTE = "oceanbench_source_kind"
DATASET_SOURCE_NAME_ATTRIBUTE = "oceanbench_source_name"
DATASET_SOURCE_RESOLUTION_ATTRIBUTE = "oceanbench_source_resolution"


@dataclass(frozen=True)
class DatasetSource:
    kind: str
    name: str
    resolution: str | None = None


def with_dataset_source(
    dataset: xarray.Dataset,
    *,
    kind: str,
    name: str,
    resolution: str | None = None,
) -> xarray.Dataset:
    attrs = dict(dataset.attrs)
    attrs[DATASET_SOURCE_KIND_ATTRIBUTE] = kind
    attrs[DATASET_SOURCE_NAME_ATTRIBUTE] = name
    if resolution is None:
        attrs.pop(DATASET_SOURCE_RESOLUTION_ATTRIBUTE, None)
    else:
        attrs[DATASET_SOURCE_RESOLUTION_ATTRIBUTE] = resolution
    return dataset.assign_attrs(attrs)


def get_dataset_source(dataset: xarray.Dataset) -> DatasetSource | None:
    kind = dataset.attrs.get(DATASET_SOURCE_KIND_ATTRIBUTE)
    name = dataset.attrs.get(DATASET_SOURCE_NAME_ATTRIBUTE)
    if kind in (None, "") or name in (None, ""):
        return None
    resolution = dataset.attrs.get(DATASET_SOURCE_RESOLUTION_ATTRIBUTE)
    return DatasetSource(kind=kind, name=name, resolution=resolution)
