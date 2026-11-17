# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import sys
from pathlib import Path

REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_DIRECTORY / "website"))

from helpers.compatible_versions import compatible_versions  # noqa: E402

CHANGELOG = """
## Unreleased

**Scores:** unchanged vs 0.6.2. Something new.

## 0.6.2 - 2026-10-29

**Scores:** unchanged vs 0.6.1. Data access only.

## 0.6.1 - 2026-10-16

**Scores:** unchanged vs 0.6.0. Fixes only.

## 0.6.0 - 2026-10-05

**Scores:** every table changes for every challenger.

## 0.5.1 - 2026-09-07

**Scores:** unchanged vs 0.5.0. Data paths only.

## 0.5.0 - 2026-08-13

**Scores:** grid-averaged scores change.
"""


def test_unchanged_releases_have_the_scores_of_the_previous_published_version():
    assert compatible_versions(CHANGELOG, ["0.5.0", "0.6.0"]) == {"0.5.1": "0.5.0", "0.6.1": "0.6.0", "0.6.2": "0.6.0"}


def test_a_release_that_changes_the_scores_has_no_published_scores_until_they_are_published():
    assert compatible_versions(CHANGELOG, ["0.5.0"]) == {"0.5.1": "0.5.0"}


def test_releases_after_a_change_of_scores_are_not_compatible_with_older_published_scores():
    assert compatible_versions(CHANGELOG, ["0.5.0", "0.5.1"]) == {}


def test_the_unreleased_changes_are_not_a_version():
    assert "Unreleased" not in compatible_versions(CHANGELOG, ["0.6.0"])
