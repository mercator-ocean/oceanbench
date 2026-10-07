# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path
import sys

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.scores_payload import scores_payload  # noqa: E402

METRIC_TITLES = {"rmsd_variables_glorys": "RMSD of variables compared to GLORYS"}
SECTIONS = {"reanalysis": {"depth_metric": "rmsd_variables_glorys", "flat_metrics": []}}


def _version_bundle(score: float) -> dict:
    return {
        "regions": {
            "global": {
                "display_name": "Global",
                "challengers": {
                    "glonet": {
                        "rmsd_variables_glorys": {
                            "name": "glonet",
                            "depths": {
                                "Surface": {
                                    "variables": {
                                        "temperature": {"unit": "°C", "data": {"1": score, "2": score * 2}},
                                    }
                                }
                            },
                        }
                    }
                },
                "challenger_names": ["glonet"],
            }
        },
        "region_order": ["global"],
    }


def _payload(version_bundles: dict, default_version: str):
    return scores_payload(version_bundles, default_version, METRIC_TITLES, SECTIONS)


def test_page_embeds_only_the_default_version_and_lists_all_versions():
    payload = _payload({"0.6.0": _version_bundle(0.5), "0.5.0": _version_bundle(0.4)}, "0.6.0")

    embedded = json.loads(payload.embedded)

    assert list(embedded["versions"]) == ["0.6.0"]
    assert embedded["version_order"] == ["0.6.0", "0.5.0"]
    assert embedded["default_version"] == "0.6.0"
    assert embedded["metric_titles"] == METRIC_TITLES
    assert embedded["sections"] == SECTIONS


def test_other_versions_are_published_separately():
    payload = _payload({"0.6.0": _version_bundle(0.5), "0.5.0": _version_bundle(0.4)}, "0.6.0")

    assert list(payload.deferred) == ["0.5.0"]
    assert json.loads(payload.deferred["0.5.0"]) == _version_bundle(0.4)


def test_scores_are_published_unchanged():
    payload = _payload({"0.6.0": _version_bundle(9.764688), "0.5.0": _version_bundle(110.239433)}, "0.6.0")

    assert json.loads(payload.embedded)["versions"]["0.6.0"] == _version_bundle(9.764688)
    assert json.loads(payload.deferred["0.5.0"]) == _version_bundle(110.239433)


def _compact(serialized: str) -> str:
    return json.dumps(json.loads(serialized), separators=(",", ":"))


def test_published_json_has_no_insignificant_whitespace():
    payload = _payload({"0.6.0": _version_bundle(0.5), "0.5.0": _version_bundle(0.4)}, "0.6.0")

    assert payload.embedded == _compact(payload.embedded)
    assert payload.deferred["0.5.0"] == _compact(payload.deferred["0.5.0"])
