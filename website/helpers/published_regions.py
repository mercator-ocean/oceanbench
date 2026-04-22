# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

GLOBAL_REGION_NAME = "global"

PUBLISHED_REGIONS = {
    GLOBAL_REGION_NAME: {
        "label": "Global",
        "description": "Global ocean domain.",
        "bounds": None,
    },
    "ibi": {
        "label": "IBI",
        "description": "Iberia-Biscay-Ireland regional domain.",
        "bounds": {
            "minimum_latitude": 26.17,
            "maximum_latitude": 56.08,
            "minimum_longitude": -19.08,
            "maximum_longitude": 5.08,
        },
    },
}


def published_region_ids() -> list[str]:
    return list(PUBLISHED_REGIONS.keys())


def published_region_label(region_id: str) -> str:
    return PUBLISHED_REGIONS[region_id]["label"]


def published_region_metadata(region_id: str) -> dict:
    return dict(PUBLISHED_REGIONS[region_id])


def published_region_ids_with_reports(published_reports: dict[str, list[str]]) -> list[str]:
    return [region_id for region_id in published_region_ids() if published_reports.get(region_id)]
