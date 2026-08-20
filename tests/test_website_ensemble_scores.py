# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path
import sys

import pandas as pd
import pytest

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.build_ensemble_scores_json import (  # noqa: E402
    build_ensemble_scores,
    with_gloens_surface,
    with_observation_sidecar,
)
from helpers.ensemble_scores import ensemble_score_bundle, ensemble_scores  # noqa: E402

GRIDDED_LEAD_DAYS = [1, 3, 5, 7, 9, 10]
OBSERVATION_LEAD_DAYS = [1, 3, 5, 7, 9, 10]


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


def _frozen_surface_frame(lead_days: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "challenger": "gloens",
                "challenger_version": "glo4-ens50_ng",
                "region": "global",
                "reference": reference,
                "variable": variable,
                "depth": depth,
                "lead_day": lead_day,
                "crps_biased": 0.1,
                "crps_fair": value,
                "ensemble_mean_rmsd": value,
                "ensemble_spread": 0.02,
                "member_rmsd": 0.16,
                "spread_error_ratio": 0.13,
                "start_count": 52,
                "scored_cells": 673289,
            }
            for reference in ("glorys", "glo12")
            for variable, depth, value in (
                ("sea_water_potential_temperature", "surface", 0.4441234),
                ("sea_surface_height_above_geoid", "surface", 0.1611234),
                ("sea_surface_height_above_geoid", "surface-datum-aligned", 0.0741234),
            )
            for lead_day in lead_days
        ]
    )


def _class4_frame(lead_days: list[int], rmsd: float = 0.8221234) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variable": "sea_water_potential_temperature",
                "depth_bin": "surface",
                "lead_day": lead_day - 1,
                "lead_day_number": lead_day,
                "rmsd": rmsd,
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
        _class4_frame(list(range(1, 11))),
        _class4_frame(list(range(1, 11)), rmsd=0.7331234),
        _class4_frame(list(range(1, 11)), rmsd=0.9441234),
        _class4_frame(list(range(1, 10)), rmsd=0.9551234),
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
    assert temperature_row["values"] == [0.881] * len(GRIDDED_LEAD_DAYS)

    ratio_row = scores["blocks"]["gridded_spread_error_ratio"]["rows"][0]
    assert ratio_row["unit"] == ""
    assert ratio_row["values"] == [0.48] * len(GRIDDED_LEAD_DAYS)


def test_build_ensemble_scores_carries_no_display_precision_of_its_own() -> None:
    scores = _built_scores()

    for block in scores["blocks"].values():
        for row in block["rows"]:
            assert "decimals" not in row


def test_build_ensemble_scores_marks_the_reduced_start_lead_days_of_gloens() -> None:
    scores = _built_scores()

    gloens_rows = [row for row in scores["blocks"]["gridded_rmsd"]["rows"] if row["system"] == "gloens"]
    icp_rows = [row for row in scores["blocks"]["gridded_rmsd"]["rows"] if row["system"] == "glonet2-ens-icp"]

    assert all(row["reduced_start_leads"] == [9, 10] for row in gloens_rows)
    assert all(row["reduced_start_leads"] == [] for row in icp_rows)
    assert all(row["values"][-1] is None for row in icp_rows)


def test_build_ensemble_scores_keeps_both_deterministic_references_next_to_the_ensemble_means() -> None:
    scores = _built_scores()

    rows = scores["blocks"]["observations_rmsd"]["rows"]

    assert [row["system"] for row in rows] == ["glonet", "glo12", "gloens", "glonet2-ens-icp"]
    assert [row["depth_band"] for row in rows] == ["Surface"] * 4
    assert [row["system_label"] for row in rows[:2]] == ["GLONET (deterministic)", "GLO12 (deterministic)"]
    assert rows[0]["values"] == [0.822] * len(OBSERVATION_LEAD_DAYS)
    assert rows[1]["values"] == [0.733] * len(OBSERVATION_LEAD_DAYS)
    assert rows[2]["values"] == [0.944] * len(OBSERVATION_LEAD_DAYS)


def test_build_ensemble_scores_takes_the_ensemble_mean_error_from_the_class4_route() -> None:
    scores = _built_scores()

    rows = scores["blocks"]["observations_rmsd"]["rows"]
    icp_row = next(row for row in rows if row["system"] == "glonet2-ens-icp")

    assert icp_row["depth_band"] == "Surface"
    assert icp_row["values"] == [0.955, 0.955, 0.955, 0.955, 0.955, None]
    assert all(row["depth_band"] in {"Surface", "15 m", "0-5 m", "5-100 m", "100-300 m", "300-600 m"} for row in rows)


def test_build_ensemble_scores_names_no_glonet2_deterministic_system() -> None:
    scores = _built_scores()

    assert "glonet2" not in scores["systems"]
    assert "glonet2" not in scores["system_order"]
    for block in scores["blocks"].values():
        assert all(row["system"] != "glonet2" for row in block["rows"])


def test_with_gloens_surface_keeps_only_the_reference_and_the_datum_aligned_sea_level() -> None:
    frame = with_gloens_surface(
        _gridded_frame("gloens", [1, 3], {}),
        _frozen_surface_frame([1, 3]),
    )
    surface = frame[frame["depth"] == "surface"]

    assert sorted(surface["variable"].unique()) == [
        "sea_surface_height_above_geoid",
        "sea_water_potential_temperature",
    ]
    assert set(surface["aggregation"]) == {"year_mean"}
    sea_level = surface[(surface["variable"] == "sea_surface_height_above_geoid") & (surface["metric"] == "crps_fair")]
    assert list(sea_level["value"].round(4).unique()) == [0.0741]
    assert set(sea_level["unit"]) == {"m"}
    ratio = surface[surface["metric"] == "spread_error_ratio"]
    assert set(ratio["unit"]) == {"1"}


def test_with_gloens_surface_refuses_an_aggregate_that_already_carries_a_surface_band() -> None:
    already = with_gloens_surface(_gridded_frame("gloens", [1, 3], {}), _frozen_surface_frame([1, 3]))

    with pytest.raises(ValueError):
        with_gloens_surface(already, _frozen_surface_frame([1, 3]))


def test_ensemble_metrics_lay_the_tables_out_like_the_deterministic_view() -> None:
    bundle = ensemble_score_bundle(_built_scores())
    sections = {section["key"]: section for section in bundle["ensemble_sections"]}

    observations = sections["ensemble-observations"]["metrics"][0]
    assert observations["unify_variables"] is False
    assert [group["depths"] for group in observations["depth_groups"]] == [
        ["0-5 m", "5-100 m", "100-300 m", "300-600 m"],
        ["Surface"],
        ["15 m"],
    ]

    gridded = sections["ensemble-gridded"]["metrics"][0]
    assert gridded["unify_variables"] is True
    assert gridded["depth_groups"] is None

    probabilistic = sections["ensemble-probabilistic"]["metrics"]
    assert [metric["unify_variables"] for metric in probabilistic] == [False, False, True, True]
    assert all(metric["depth_groups"] is None for metric in probabilistic)


def test_with_observation_sidecar_adds_the_streams_of_a_later_campaign_wave() -> None:
    frame = with_observation_sidecar(
        _observation_frame([1, 3], ["drifter_sst"]),
        _observation_frame([1, 3], ["currents_u"]),
    )

    assert sorted(frame["stream"].unique()) == ["currents_u", "drifter_sst"]
    assert len(frame) == 8


def test_with_observation_sidecar_refuses_a_sidecar_repeating_the_aggregate() -> None:
    with pytest.raises(ValueError):
        with_observation_sidecar(
            _observation_frame([1, 3], ["drifter_sst", "currents_u"]),
            _observation_frame([1, 3], ["currents_u"]),
        )


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
    assert region["challenger_names"] == ["glonet", "glo12", "gloens", "glonet2-ens-icp"]
    assert bundle["versions"]["ensemble"]["challenger_labels"]["gloens"] == "GloEns"
    assert bundle["versions"]["ensemble"]["challenger_labels"]["glo12"] == "GLO12 (deterministic)"

    temperature = region["challengers"]["gloens"]["rmsd_gridded"]["depths"]["47.374 m"]["variables"]["Temperature"]
    assert temperature["unit"] == "°C"
    assert "decimals" not in temperature
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
    assert [metric["colorize"] for metric in sections[2]["metrics"]] == [True, True, True, True]
    assert [metric["color_transform"] for metric in sections[2]["metrics"]] == [
        None,
        "closeness_to_one",
        None,
        "closeness_to_one",
    ]
    ratio_note = sections[2]["metrics"][1]["note"]
    assert "shaded by closeness to one" in ratio_note

    observations = sections[0]["metrics"][0]
    layout_depths = list(observations["layout"]["depths"])
    challengers = bundle["versions"]["ensemble"]["regions"]["global"]["challengers"]
    for system_scores in challengers.values():
        assert set(system_scores["rmsd_observations"]["depths"]) <= set(layout_depths)
