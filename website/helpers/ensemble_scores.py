# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
import os

from helpers.published_regions import GLOBAL_REGION_NAME, published_region_label, published_region_metadata

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ENSEMBLE_SCORES_PATH = os.path.join(os.path.dirname(SCRIPT_DIRECTORY), "data", "ensemble-scores.json")

ENSEMBLE_VERSION_NAME = "ensemble"
SURFACE_DEPTH_BAND = "Surface"

METRIC_KEYS = {
    "observations_rmsd": "rmsd_observations",
    "gridded_rmsd": "rmsd_gridded",
    "observations_crps": "crps_observations",
    "observations_spread_error_ratio": "spread_error_ratio_observations",
    "gridded_crps": "crps_gridded",
    "gridded_spread_error_ratio": "spread_error_ratio_gridded",
}

COLORED_BLOCKS = {"observations_rmsd", "gridded_rmsd", "observations_crps", "gridded_crps"}

REDUCED_START_NOTE = "Lead days 9 and 10 of GloEns average 51 starts instead of 52."
UNCOLORED_NOTE = "Cells are not colored, as one is the target of the ratio and not a best value."

BLOCK_NOTES = {
    "observations_rmsd": (
        "The deterministic rows come from the class 4 matchup and keep its own depth bins, the ensemble rows come "
        "from the superob matchup, so the rows of a variable are not paired one to one. "
        "GloEns has no current observations."
    ),
    "gridded_rmsd": f"GloEns carries no surface fields and GloNet2-ens-icp stops at lead day 9. {REDUCED_START_NOTE}",
    "gridded_crps": REDUCED_START_NOTE,
    "gridded_spread_error_ratio": REDUCED_START_NOTE,
}

SECTIONS = [
    {
        "key": "ensemble-observations",
        "label": "Observations",
        "container": "ensemble-observations-scores",
        "blocks": ["observations_rmsd"],
    },
    {
        "key": "ensemble-gridded",
        "label": "GLORYS",
        "container": "ensemble-gridded-scores",
        "blocks": ["gridded_rmsd"],
    },
    {
        "key": "ensemble-probabilistic",
        "label": "Probabilistic",
        "container": "ensemble-probabilistic-scores",
        "blocks": [
            "observations_crps",
            "observations_spread_error_ratio",
            "gridded_crps",
            "gridded_spread_error_ratio",
        ],
    },
]


def ensemble_scores(path: str = ENSEMBLE_SCORES_PATH) -> dict:
    """Read the ensemble scores committed next to the page by build_ensemble_scores_json.py."""
    with open(path) as file:
        return json.load(file)


def _ordered_unique(values) -> list:
    ordered = []
    for value in values:
        if value not in ordered:
            ordered.append(value)
    return ordered


def _variable_order(block: dict) -> list[str]:
    return _ordered_unique(row["variable"] for row in block["rows"])


def _depth_order(block: dict) -> list[str]:
    depths = _ordered_unique(row["depth_band"] for row in block["rows"])
    return [depth for depth in depths if depth == SURFACE_DEPTH_BAND] + [
        depth for depth in depths if depth != SURFACE_DEPTH_BAND
    ]


def _depth_score(rows: list[dict], variable_order: list[str], lead_days: list[int]) -> dict:
    return {
        "variables": {
            row["variable"]: {
                "unit": row["unit"],
                "standard_name": "",
                "decimals": row["decimals"],
                "data": {str(lead_day): value for lead_day, value in zip(lead_days, row["values"])},
            }
            for row in sorted(rows, key=lambda row: variable_order.index(row["variable"]))
        }
    }


def _score(block: dict, name: str, rows: list[dict]) -> dict:
    variable_order = _variable_order(block)
    depth_rows = {depth: [row for row in rows if row["depth_band"] == depth] for depth in _depth_order(block)}
    return {
        "name": name,
        "depths": {
            depth: _depth_score(depth_group, variable_order, block["lead_days"])
            for depth, depth_group in depth_rows.items()
            if depth_group
        },
    }


def _system_score(block: dict, system: str) -> dict | None:
    system_rows = [row for row in block["rows"] if row["system"] == system]
    if not system_rows:
        return None
    return _score(block, system, system_rows)


def _layout_score(block: dict) -> dict:
    """Every depth and variable of a block, so a table shows the rows of systems the baseline does not cover."""
    return _score(block, "layout", block["rows"])


def _challenger_scores(scores: dict) -> dict:
    return {
        system: {
            METRIC_KEYS[block_key]: score
            for block_key, block in scores["blocks"].items()
            if (score := _system_score(block, system))
        }
        for system in scores["system_order"]
    }


def _metric_note(block_key: str, block: dict) -> str:
    notes = [block["note"], BLOCK_NOTES.get(block_key, "")]
    if block_key not in COLORED_BLOCKS:
        notes.append(UNCOLORED_NOTE)
    return " ".join(note for note in notes if note)


def _section_metrics(scores: dict, block_keys: list[str]) -> list[dict]:
    return [
        {
            "metric_key": METRIC_KEYS[block_key],
            "title": scores["blocks"][block_key]["title"],
            "note": _metric_note(block_key, scores["blocks"][block_key]),
            "colorize": block_key in COLORED_BLOCKS,
            "layout": _layout_score(scores["blocks"][block_key]),
        }
        for block_key in block_keys
    ]


def ensemble_score_bundle(scores: dict | None = None) -> dict:
    """Shape the committed ensemble scores like the score bundle the deterministic tables are built from."""
    resolved_scores = scores if scores is not None else ensemble_scores()
    challengers = _challenger_scores(resolved_scores)
    return {
        "versions": {
            ENSEMBLE_VERSION_NAME: {
                "regions": {
                    GLOBAL_REGION_NAME: {
                        "display_name": published_region_label(GLOBAL_REGION_NAME),
                        "challengers": challengers,
                        "challenger_names": list(challengers),
                    }
                },
                "region_order": [GLOBAL_REGION_NAME],
                "region_labels": {GLOBAL_REGION_NAME: published_region_label(GLOBAL_REGION_NAME)},
                "region_metadata": {GLOBAL_REGION_NAME: published_region_metadata(GLOBAL_REGION_NAME)},
                "challenger_labels": {system: resolved_scores["systems"][system]["label"] for system in challengers},
            }
        },
        "version_order": [ENSEMBLE_VERSION_NAME],
        "default_version": ENSEMBLE_VERSION_NAME,
        "sections": {},
        "metric_titles": {},
        "ensemble_sections": [
            {
                "key": section["key"],
                "label": section["label"],
                "container": section["container"],
                "metrics": _section_metrics(resolved_scores, section["blocks"]),
            }
            for section in SECTIONS
        ],
    }
