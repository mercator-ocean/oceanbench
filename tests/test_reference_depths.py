# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

from oceanbench.core.reference_depths import (
    REFERENCE_DEPTH_GRID_HASH_ATTRIBUTE,
    REFERENCE_DEPTH_GRID_ROUNDING_ATTRIBUTE,
    REFERENCE_DEPTH_GRID_ROUNDING_DECIMALS,
    REFERENCE_TARGET_DEPTHS_ATTRIBUTE,
    reference_depth_grid_variant,
    with_reference_depth_grid_metadata,
)
from oceanbench.core.references import glo12, glorys


def test_reference_depth_grid_variant_uses_rounded_depths() -> None:
    xihe_depths = numpy.array([0.494, 2.6457, 5.0782, 7.9296])
    wenhai_depths = numpy.array([0.494025, 2.645669, 5.078224, 7.929560])
    glo12_depths = numpy.array([0.494025, 47.37369, 92.32607, 155.8507])

    assert reference_depth_grid_variant(xihe_depths) == reference_depth_grid_variant(wenhai_depths)
    assert reference_depth_grid_variant(xihe_depths) != reference_depth_grid_variant(glo12_depths)


def test_with_reference_depth_grid_metadata_records_auditable_depth_grid() -> None:
    target_depths = numpy.array([0.494025, 2.645669])
    dataset = xarray.Dataset()

    dataset_with_metadata = with_reference_depth_grid_metadata(dataset, target_depths)

    assert dataset_with_metadata.attrs[REFERENCE_DEPTH_GRID_HASH_ATTRIBUTE]
    assert (
        dataset_with_metadata.attrs[REFERENCE_DEPTH_GRID_ROUNDING_ATTRIBUTE] == REFERENCE_DEPTH_GRID_ROUNDING_DECIMALS
    )
    assert dataset_with_metadata.attrs[REFERENCE_TARGET_DEPTHS_ATTRIBUTE] == [0.494, 2.646]


def test_glorys_twelfth_degree_cache_key_depends_on_target_depth_grid() -> None:
    first_day_datetime = numpy.datetime64("2024-01-03")
    xihe_variant = reference_depth_grid_variant(numpy.array([0.494, 2.6457]))
    wenhai_variant = reference_depth_grid_variant(numpy.array([0.494025, 2.645669]))
    glo12_variant = reference_depth_grid_variant(numpy.array([0.494025, 47.37369]))

    def cache_key(depth_grid_variant: str) -> str:
        return glorys._glorys_1_12_week_cache_key(first_day_datetime, 10, depth_grid_variant)

    assert cache_key(xihe_variant) == cache_key(wenhai_variant)
    assert cache_key(xihe_variant) != cache_key(glo12_variant)
    assert "twelfth_degree" in cache_key(xihe_variant)


def test_glo12_twelfth_degree_cache_key_depends_on_target_depth_grid() -> None:
    first_day_datetime = numpy.datetime64("2024-01-03")
    xihe_variant = reference_depth_grid_variant(numpy.array([0.494, 2.6457]))
    wenhai_variant = reference_depth_grid_variant(numpy.array([0.494025, 2.645669]))
    glo12_variant = reference_depth_grid_variant(numpy.array([0.494025, 47.37369]))

    def cache_key(depth_grid_variant: str) -> str:
        return glo12._glo12_1_12_week_cache_key(first_day_datetime, 10, depth_grid_variant)

    assert cache_key(xihe_variant) == cache_key(wenhai_variant)
    assert cache_key(xihe_variant) != cache_key(glo12_variant)
    assert "twelfth_degree" in cache_key(xihe_variant)
