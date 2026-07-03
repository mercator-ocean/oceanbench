# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
import pytest

from oceanbench.runner import parity


def test_golden_metric_key_mapping():
    assert parity.golden_metric_key("rmsd", "glorys", "sea_surface_height_above_geoid") == "rmsd_variables_glorys"
    assert parity.golden_metric_key("rmsd", "glo12", "sea_water_salinity") == "rmsd_variables_glo12"
    assert parity.golden_metric_key("rmsd", "glorys", "ocean_mixed_layer_thickness") == "rmsd_mld_glorys"
    assert (
        parity.golden_metric_key("rmsd", "glo12", "geostrophic_eastward_sea_water_velocity") == "rmsd_geostrophic_glo12"
    )
    assert (
        parity.golden_metric_key("class4_rmsd", "observations", "sea_water_salinity") == "rmsd_variables_observations"
    )
    assert parity.golden_metric_key("lagrangian_deviation_km", "glorys", None) == "lagrangian_glorys"


def _runner_row(**overrides) -> dict:
    row = {
        "challenger": "glonet_1_degree",
        "challenger_version": "0.2.1",
        "year": 2024,
        "region": "global",
        "metric": "rmsd",
        "reference": "glorys",
        "variable": "sea_surface_height_above_geoid",
        "depth": "surface",
        "lead_day": 1,
        "start_date": None,
        "band": None,
        "polarity": None,
        "value": 0.0,
        "unit": "m",
        "n": None,
        "oceanbench_version": "0.2.1",
    }
    row.update(overrides)
    return row


def _golden_row(**overrides) -> dict:
    row = {
        "source_version": "0.2.1",
        "challenger": "glonet_1_degree",
        "region": "global",
        "metric_key": "rmsd_variables_glorys",
        "variable_standard_name": "sea_surface_height_above_geoid",
        "display_name": "sea surface height",
        "unit": "m",
        "depth_label": "Surface",
        "lead_day": 1,
        "value": 0.0,
    }
    row.update(overrides)
    return row


def test_per_start_mean_matches_golden_within_tolerance():
    # Two start dates whose mean is exactly the golden value.
    runner = pandas.DataFrame(
        [
            _runner_row(start_date="2024-01-03", value=0.06),
            _runner_row(start_date="2024-01-10", value=0.08),
        ]
    )
    golden = pandas.DataFrame([_golden_row(value=0.07)])
    comparisons = parity.compare(runner, golden)
    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison.golden_metric_key == "rmsd_variables_glorys"
    assert comparison.matched == 1
    assert comparison.max_absolute_difference == pytest.approx(0.0, abs=1e-12)
    passed, failures = parity.gate(comparisons)
    assert passed and failures == []


def test_gate_flags_out_of_tolerance():
    runner = pandas.DataFrame([_runner_row(start_date="2024-01-03", value=0.10)])
    golden = pandas.DataFrame([_golden_row(value=0.07)])
    comparisons = parity.compare(runner, golden)
    assert comparisons[0].max_absolute_difference == pytest.approx(0.03)
    passed, failures = parity.gate(comparisons, absolute_tolerance=1e-4)
    assert not passed
    assert "rmsd_variables_glorys" in failures[0]


def test_depth_label_case_and_surface_align():
    runner = pandas.DataFrame([_runner_row(start_date="2024-01-03", value=0.07, depth="surface")])
    golden = pandas.DataFrame([_golden_row(value=0.07, depth_label="Surface")])
    comparison = parity.compare(runner, golden)[0]
    assert comparison.matched == 1
    assert comparison.golden_only == 0 and comparison.runner_only == 0


def test_null_depth_and_variable_keys_join():
    # Lagrangian: variable and depth both null on each side must still join.
    runner = pandas.DataFrame(
        [
            _runner_row(
                metric="lagrangian_deviation_km",
                reference="glorys",
                variable=None,
                depth=None,
                unit="km",
                lead_day=2,
                value=15.0,
                start_date="2024-01-03",
            )
        ]
    )
    golden = pandas.DataFrame(
        [
            _golden_row(
                metric_key="lagrangian_glorys",
                variable_standard_name="",
                depth_label=None,
                unit="km",
                lead_day=2,
                value=15.0,
            )
        ]
    )
    comparison = parity.compare(runner, golden)[0]
    assert comparison.golden_metric_key == "lagrangian_glorys"
    assert comparison.matched == 1
    assert comparison.runner_only == 0 and comparison.golden_only == 0


def test_exclude_metric_is_skipped():
    runner = pandas.DataFrame(
        [
            _runner_row(
                metric="lagrangian_deviation_km", reference="glorys", variable=None, depth=None, lead_day=2, value=1.0
            )
        ]
    )
    golden = pandas.DataFrame(
        [
            _golden_row(
                metric_key="lagrangian_glorys", variable_standard_name="", depth_label=None, lead_day=2, value=9.0
            )
        ]
    )
    comparisons = parity.compare(runner, golden, exclude_golden_metrics=("lagrangian_glorys",))
    assert comparisons == []
