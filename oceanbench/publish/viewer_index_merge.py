# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Merge the viewer index files of several challenger and region evaluations into one set of index files.

Each evaluation writes ``datasets.json``, ``insights.json`` and ``scores-summary.json`` for its own challenger and
region only. These functions combine such fragments, so evaluations that ran on different machines can be published
together.
"""

from functools import reduce


def summary_sort_key(row: dict) -> tuple:
    return (
        row["challenger"],
        row["year"],
        row["region"],
        row["metric"],
        row["reference"],
        row["variable"],
        row["depth"] is None,
        row["depth"] or "",
        row["lead_day"],
    )


def merge_datasets(fragments: list[dict]) -> dict:
    entries = {entry["slug"]: entry for fragment in fragments for entry in fragment["datasets"]}
    return {"datasets": [entries[slug] for slug in sorted(entries)]}


def _merge_dataset_insights(merged: dict, fragment_datasets: dict) -> dict:
    return {
        slug: {**merged.get(slug, {}), **fragment_datasets.get(slug, {})}
        for slug in sorted({*merged, *fragment_datasets})
    }


def merge_insights(fragments: list[dict]) -> dict:
    return {
        "datasets": reduce(_merge_dataset_insights, (fragment["datasets"] for fragment in fragments), {}),
        "regions": {region: bounds for fragment in fragments for region, bounds in fragment["regions"].items()},
        "scores_summary": fragments[-1]["scores_summary"],
    }


def merge_scores_summary(fragments: list[list[dict]]) -> list[dict]:
    return sorted((row for fragment in fragments for row in fragment), key=summary_sort_key)
