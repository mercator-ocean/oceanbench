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

# A ratio has no "lower is better" direction, so its cells are shaded by how far they sit from
# one instead of by their value. The shading then reads on the same ramp as every other table.
CLOSENESS_TO_ONE_BLOCKS = {"observations_spread_error_ratio", "gridded_spread_error_ratio"}

REDUCED_START_NOTE = "Lead days 9 and 10 of GloEns average 51 starts instead of 52."
DATUM_NOTE = "The sea surface height of GloEns is datum aligned to the reference before it is scored."
SHORT_HORIZON_NOTE = "GloNet2-ens-icp stops at lead day 9."
CLOSENESS_TO_ONE_NOTE = (
    "Cells are shaded by closeness to one, the target of the ratio: a cell reads better than the "
    "baseline when its ratio sits nearer to one, whichever side of one it falls. The percentages "
    "the cells and the tooltips carry compare those same distances from one, not the ratios "
    "themselves, so a cell never reads better in colour and worse in number."
)

# The error table reads one matchup for everybody, the probabilistic tables need the members and so
# read the superob matchup, which is a different pairing of forecast and observation.
SUPEROB_MATCHUP_NOTE = (
    "These scores come from the superob matchup, where observations are placed on the quarter "
    "degree cells of the model and averaged when several of them share a cell, because an ensemble "
    "predicts the average of a cell while a single observation also carries the small scale noise "
    "no forecast can resolve. In practice that averaging bites on the along track sea level, which "
    "brings about seven observations to a cell; the drifter temperature and the currents arrive at "
    "roughly one observation per cell, so for them the step is a change of position and not an "
    "averaging away of noise."
)

BLOCK_NOTES = {
    "observations_rmsd": (
        "The ensemble mean of each ensemble is scored exactly as the deterministic systems are, through the "
        "class 4 matchup and its depth bins, so every row of a variable is directly comparable. "
        f"{SHORT_HORIZON_NOTE}"
    ),
    "observations_crps": f"{SHORT_HORIZON_NOTE} {SUPEROB_MATCHUP_NOTE}",
    "observations_spread_error_ratio": f"{SHORT_HORIZON_NOTE} {SUPEROB_MATCHUP_NOTE}",
    "gridded_rmsd": f"{DATUM_NOTE} {SHORT_HORIZON_NOTE} {REDUCED_START_NOTE}",
    "gridded_crps": f"{DATUM_NOTE} {SHORT_HORIZON_NOTE} {REDUCED_START_NOTE}",
    "gridded_spread_error_ratio": f"{DATUM_NOTE} {SHORT_HORIZON_NOTE} {REDUCED_START_NOTE}",
}

# The error table now carries the class 4 depth bins for every system, so it is laid out in the
# three tables the deterministic view lays them out in rather than in whatever groups the depths
# happen to fall into. The probabilistic tables keep the superob bands and are left to group
# themselves.
OBSERVATION_DEPTH_GROUPS = [
    {
        "depths": ["0-5 m", "5-100 m", "100-300 m", "300-600 m"],
        "variables": ["Profile temperature", "Profile salinity"],
    },
    {
        "depths": ["Surface"],
        "variables": ["Drifter SST", "Sea level anomaly"],
        "show_depth_label": True,
    },
    {
        "depths": ["15 m"],
        "variables": ["Eastward current", "Northward current"],
        "show_depth_label": True,
    },
]

# The gridded tables run over one depth axis with sea level living only at its surface, exactly as
# the deterministic gridded tables do, so they unify their variables into a single table and leave
# a blank where a depth does not carry a variable.
UNIFIED_VARIABLE_BLOCKS = {"gridded_rmsd", "gridded_crps", "gridded_spread_error_ratio"}
BLOCK_DEPTH_GROUPS = {"observations_rmsd": OBSERVATION_DEPTH_GROUPS}

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
    if block_key in CLOSENESS_TO_ONE_BLOCKS:
        notes.append(CLOSENESS_TO_ONE_NOTE)
    return " ".join(note for note in notes if note)


def _section_metrics(scores: dict, block_keys: list[str]) -> list[dict]:
    return [
        {
            "metric_key": METRIC_KEYS[block_key],
            "title": scores["blocks"][block_key]["title"],
            "note": _metric_note(block_key, scores["blocks"][block_key]),
            "color_transform": "closeness_to_one" if block_key in CLOSENESS_TO_ONE_BLOCKS else None,
            "unify_variables": block_key in UNIFIED_VARIABLE_BLOCKS,
            "depth_groups": BLOCK_DEPTH_GROUPS.get(block_key),
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
