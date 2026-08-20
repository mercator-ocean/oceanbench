# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
import os

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ENSEMBLE_SCORES_PATH = os.path.join(os.path.dirname(SCRIPT_DIRECTORY), "data", "ensemble-scores.json")


def ensemble_scores(path: str = ENSEMBLE_SCORES_PATH) -> dict:
    """Read the ensemble scores committed next to the page by build_ensemble_scores_json.py."""
    with open(path) as file:
        return json.load(file)
