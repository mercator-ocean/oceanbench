# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from dataclasses import dataclass
from typing import Any

SCORES_DIRECTORY = "scores"


@dataclass(frozen=True)
class ScoresPayload:
    embedded: str
    deferred: dict[str, str]


def _compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def scores_payload(
    version_bundles: dict[str, dict],
    default_version: str,
    metric_titles: dict[str, str],
    sections: dict[str, dict],
) -> ScoresPayload:
    """Split the scores into the page payload, holding only the default version, and one payload per other version."""
    embedded_versions = {version: bundle for version, bundle in version_bundles.items() if version == default_version}
    return ScoresPayload(
        embedded=_compact_json(
            {
                "versions": embedded_versions,
                "version_order": list(version_bundles),
                "default_version": default_version,
                "scores_directory": SCORES_DIRECTORY,
                "metric_titles": metric_titles,
                "sections": sections,
            }
        ),
        deferred={
            version: _compact_json(bundle) for version, bundle in version_bundles.items() if version != default_version
        },
    )


def scores_file_path(version: str) -> str:
    """Relative path of the published scores of a version that is not embedded in the page."""
    return f"{SCORES_DIRECTORY}/{version}.json"
