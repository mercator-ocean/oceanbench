# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path
import sys

import pandas as pd

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.build_ensemble_scores_json import build_ensemble_scores  # noqa: E402
from helpers.ensemble_scores import ensemble_score_bundle, ensemble_scores  # noqa: E402

GRIDDED_LEAD_DAYS = [1, 3, 5, 7, 9, 10]
OBSERVATION_LEAD_DAYS = [1, 3, 5, 7, 9]


def _gridded_frame(challenger: str, lead_days: list[int], start_counts: dict[int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "aggregation": "year_mean",
                "challenger": challenger,
                "region": "global",
                "reference": "glorys",
                "variable": "sea_water_potential_temperature",
                "depth": "47.374m",
                "lead_day": lead_day,
                "metric": metric,
                "value": value,
                "unit": "°C",
                "start_count": start_counts.get(lead_day, 52),
            }
            for lead_day in lead_days
            for metric, value in (
                ("ensemble_mean_rmsd", 0.8812345),
                ("crps_fair", 0.4481234),
                ("spread_error_ratio", 0.4812345),
            )
        ]
    )


def _observation_frame(lead_days: list[int], streams: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stream": stream,
                "region": region,
                "depth_band": "all",
                "lead_day": lead_day,
                "n_inits": 52,
                "crps_fair": 0.3211234,
                "ssr_add": 0.3412345,
                "rmsd_ensemble_mean": 0.8521234,
            }
            for stream in streams
            for region in ("global", "tropics")
            for lead_day in lead_days
        ]
    )


def _deterministic_frame(lead_days: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variable": "sea_water_potential_temperature",
                "depth_bin": "surface",
                "lead_day": lead_day - 1,
                "lead_day_number": lead_day,
                "rmsd": 0.8221234,
                "observation_count": 1000,
                "start_count": 52,
            }
            for lead_day in lead_days
        ]
    )


def _built_scores() -> dict:
    return build_ensemble_scores(
        _gridded_frame("gloens", list(range(1, 11)), {9: 51, 10: 51}),
        _gridded_frame("glonet2-ens-icp", list(range(1, 10)), {}),
        _deterministic_frame(list(range(1, 10))),
        _observation_frame(list(range(1, 11)), ["drifter_sst"]),
        _observation_frame(list(range(1, 10)), ["drifter_sst"]),
    )


def test_build_ensemble_scores_exposes_every_block_on_the_expected_lead_days() -> None:
    scores = _built_scores()

    assert list(scores["blocks"]) == [
        "observations_rmsd",
        "gridded_rmsd",
        "gridded_crps",
        "gridded_spread_error_ratio",
        "observations_crps",
        "observations_spread_error_ratio",
    ]
    for block_key, block in scores["blocks"].items():
        expected_lead_days = GRIDDED_LEAD_DAYS if block_key.startswith("gridded") else OBSERVATION_LEAD_DAYS
        assert block["lead_days"] == expected_lead_days
        for row in block["rows"]:
            assert len(row["values"]) == len(expected_lead_days)


def test_build_ensemble_scores_rounds_and_labels_each_row() -> None:
    scores = _built_scores()

    temperature_row = scores["blocks"]["gridded_rmsd"]["rows"][0]
    assert temperature_row["system"] == "gloens"
    assert temperature_row["variable"] == "Temperature"
    assert temperature_row["depth_band"] == "47.374 m"
    assert temperature_row["unit"] == "°C"
    assert temperature_row["decimals"] == 3
    assert temperature_row["values"] == [0.881] * len(GRIDDED_LEAD_DAYS)

    ratio_row = scores["blocks"]["gridded_spread_error_ratio"]["rows"][0]
    assert ratio_row["unit"] == ""
    assert ratio_row["decimals"] == 2
    assert ratio_row["values"] == [0.48] * len(GRIDDED_LEAD_DAYS)


def test_build_ensemble_scores_marks_the_reduced_start_lead_days_of_gloens() -> None:
    scores = _built_scores()

    gloens_rows = [row for row in scores["blocks"]["gridded_rmsd"]["rows"] if row["system"] == "gloens"]
    icp_rows = [row for row in scores["blocks"]["gridded_rmsd"]["rows"] if row["system"] == "glonet2-ens-icp"]

    assert all(row["reduced_start_leads"] == [9, 10] for row in gloens_rows)
    assert all(row["reduced_start_leads"] == [] for row in icp_rows)
    assert all(row["values"][-1] is None for row in icp_rows)


def test_build_ensemble_scores_keeps_the_deterministic_baseline_next_to_the_ensemble_means() -> None:
    scores = _built_scores()

    rows = scores["blocks"]["observations_rmsd"]["rows"]

    assert [row["system"] for row in rows] == ["glonet2", "gloens", "glonet2-ens-icp"]
    assert [row["depth_band"] for row in rows] == ["Surface", "Surface", "Surface"]
    assert rows[0]["values"] == [0.822] * len(OBSERVATION_LEAD_DAYS)
    assert rows[1]["values"] == [0.852] * len(OBSERVATION_LEAD_DAYS)


def test_build_ensemble_scores_reads_the_global_region_only() -> None:
    scores = _built_scores()

    assert len(scores["blocks"]["observations_crps"]["rows"]) == 2


def test_committed_ensemble_scores_match_the_converter_contract() -> None:
    scores = ensemble_scores()

    assert scores["year"] == 2024
    assert set(scores["system_order"]) == set(scores["systems"])
    for block in scores["blocks"].values():
        assert block["rows"]
        for row in block["rows"]:
            assert len(row["values"]) == len(block["lead_days"])
            assert row["system"] in scores["systems"]


def test_ensemble_score_bundle_reads_like_the_deterministic_score_bundle() -> None:
    bundle = ensemble_score_bundle(_built_scores())
    region = bundle["versions"]["ensemble"]["regions"]["global"]

    assert bundle["version_order"] == ["ensemble"]
    assert bundle["default_version"] == "ensemble"
    assert region["display_name"] == "Global"
    assert region["challenger_names"] == ["glonet2", "gloens", "glonet2-ens-icp"]
    assert bundle["versions"]["ensemble"]["challenger_labels"]["gloens"] == "GloEns"

    temperature = region["challengers"]["gloens"]["rmsd_gridded"]["depths"]["47.374 m"]["variables"]["Temperature"]
    assert temperature["unit"] == "°C"
    assert temperature["decimals"] == 3
    assert temperature["data"] == {str(lead_day): 0.881 for lead_day in GRIDDED_LEAD_DAYS}


def test_ensemble_score_bundle_exposes_a_metric_per_section_with_a_full_layout() -> None:
    bundle = ensemble_score_bundle(_built_scores())

    sections = bundle["ensemble_sections"]
    assert [section["key"] for section in sections] == [
        "ensemble-observations",
        "ensemble-gridded",
        "ensemble-probabilistic",
    ]
    assert [metric["metric_key"] for metric in sections[2]["metrics"]] == [
        "crps_observations",
        "spread_error_ratio_observations",
        "crps_gridded",
        "spread_error_ratio_gridded",
    ]
    assert [metric["colorize"] for metric in sections[2]["metrics"]] == [True, False, True, False]

    observations = sections[0]["metrics"][0]
    layout_depths = list(observations["layout"]["depths"])
    challengers = bundle["versions"]["ensemble"]["regions"]["global"]["challengers"]
    for system_scores in challengers.values():
        assert set(system_scores["rmsd_observations"]["depths"]) <= set(layout_depths)
