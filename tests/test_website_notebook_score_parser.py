# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBSITE_DIRECTORY = PROJECT_ROOT / "website"
if str(WEBSITE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.notebook_score_parser import get_model_score_from_notebook  # noqa: E402


NOTEBOOK_PATH = str(PROJECT_ROOT / "assets" / "glonet_sample.report.ipynb")


def test_parse_glorys_reanalysis_score_from_notebook():
    score = get_model_score_from_notebook(NOTEBOOK_PATH, "GLONET", "glorys_reanalysis")

    assert score.name == "GLONET"
    assert "Surface" in score.depths
    assert "temperature" in score.depths["Surface"].variables
    assert "mixed layer depth" in score.depths["Surface"].variables
    assert "northward geostrophic velocity" in score.depths["Surface"].variables


def test_parse_observation_score_from_notebook():
    score = get_model_score_from_notebook(NOTEBOOK_PATH, "GLONET", "observations")

    assert score.name == "GLONET"
    assert "Surface" in score.depths
    assert "surface temperature" in score.depths["Surface"].variables
    assert "0-5m" in score.depths
