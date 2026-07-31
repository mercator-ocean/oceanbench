# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path
import sys

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.published_regions import PUBLISHED_REGIONS  # noqa: E402

from oceanbench.core.regions import OFFICIAL_REGIONS, BoundingBox  # noqa: E402


def _bounds_mapping(bounds: BoundingBox | None) -> dict[str, float] | None:
    if bounds is None:
        return None
    return {
        "minimum_latitude": bounds.minimum_latitude,
        "maximum_latitude": bounds.maximum_latitude,
        "minimum_longitude": bounds.minimum_longitude,
        "maximum_longitude": bounds.maximum_longitude,
    }


def test_published_regions_match_official_region_ids() -> None:
    assert set(PUBLISHED_REGIONS) == set(OFFICIAL_REGIONS)


def test_published_regions_match_official_region_bounds() -> None:
    for region_id, official_region in OFFICIAL_REGIONS.items():
        assert PUBLISHED_REGIONS[region_id]["bounds"] == _bounds_mapping(official_region.bounds)


def test_published_regions_carry_the_expected_bounds() -> None:
    assert PUBLISHED_REGIONS["global"]["bounds"] is None
    assert PUBLISHED_REGIONS["ibi"]["bounds"] == {
        "minimum_latitude": 26.17,
        "maximum_latitude": 56.08,
        "minimum_longitude": -19.08,
        "maximum_longitude": 5.08,
    }
