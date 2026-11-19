# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import re
from functools import reduce

from helpers.s3_discovery import version_sort_key

RELEASE_SCORES_PATTERN = re.compile(r"^## (\d+\.\d+\.\d+) - \S+\s+\*\*Scores:\*\* (.*)$", re.MULTILINE)
UNCHANGED_SCORES_PREFIX = "unchanged"


def _released_versions(changelog: str) -> list[tuple[str, bool]]:
    """Each released version of the changelog, oldest first, with whether its scores are unchanged."""
    return sorted(
        (
            (version, scores_description.lower().startswith(UNCHANGED_SCORES_PREFIX))
            for version, scores_description in RELEASE_SCORES_PATTERN.findall(changelog)
        ),
        key=lambda released_version: version_sort_key(released_version[0]),
    )


def _scores_version(
    published_versions: list[str], previous_scores_version: str | None, version: str, scores_unchanged: bool
) -> str | None:
    if version in published_versions:
        return version
    return previous_scores_version if scores_unchanged else None


def compatible_versions(changelog: str, published_versions: list[str]) -> dict[str, str]:
    """The published version whose scores each unpublished released version computes, as its changelog states it.

    A release has the scores of the previous release when its changelog entry says its scores are unchanged.
    """

    def add_release(
        state: tuple[dict[str, str], str | None], released_version: tuple[str, bool]
    ) -> tuple[dict[str, str], str | None]:
        compatible, previous_scores_version = state
        version, scores_unchanged = released_version
        scores_version = _scores_version(published_versions, previous_scores_version, version, scores_unchanged)
        unpublished_match = {version: scores_version} if scores_version and version not in published_versions else {}
        return {**compatible, **unpublished_match}, scores_version

    return reduce(add_release, _released_versions(changelog), ({}, None))[0]
