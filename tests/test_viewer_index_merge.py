# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from oceanbench.publish.viewer_index_merge import merge_datasets, merge_insights, merge_scores_summary


def _dataset(slug: str) -> dict:
    return {"slug": slug, "label": slug, "store": f"./data/{slug}.zarr"}


def _insights(slug: str, region: str, bounds) -> dict:
    return {
        "datasets": {slug: {region: {"class4_matchups": f"./data/insights/{slug}/{region}/class4-matchups.parquet"}}},
        "regions": {region: bounds},
        "scores_summary": "./data/scores-summary.json",
    }


def _row(challenger: str, region: str, lead_day: int) -> dict:
    return {
        "challenger": challenger,
        "year": 2024,
        "region": region,
        "metric": "rmsd_of_variables",
        "reference": "glorys",
        "variable": "temperature",
        "depth": "surface",
        "lead_day": lead_day,
    }


def test_the_datasets_of_every_fragment_are_listed_once_in_slug_order():
    merged = merge_datasets(
        [{"datasets": [_dataset("xihe")]}, {"datasets": [_dataset("glonet"), _dataset("xihe")]}]
    )

    assert [entry["slug"] for entry in merged["datasets"]] == ["glonet", "xihe"]


def test_the_insights_of_a_challenger_gather_its_regions_across_fragments():
    ibi_bounds = {"minimum_latitude": 26.0}

    merged = merge_insights(
        [_insights("glonet", "global", None), _insights("glonet", "ibi", ibi_bounds), _insights("xihe", "global", None)]
    )

    assert sorted(merged["datasets"]["glonet"]) == ["global", "ibi"]
    assert sorted(merged["datasets"]) == ["glonet", "xihe"]
    assert merged["regions"] == {"global": None, "ibi": ibi_bounds}


def test_the_scores_summary_rows_of_every_fragment_are_sorted_together():
    merged = merge_scores_summary(
        [[_row("xihe", "global", 1)], [_row("glonet", "ibi", 2), _row("glonet", "ibi", 1)]]
    )

    assert [(row["challenger"], row["lead_day"]) for row in merged] == [("glonet", 1), ("glonet", 2), ("xihe", 1)]
