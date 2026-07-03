# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import math

import pandas
import pytest

from oceanbench.runner import records


@pytest.fixture
def context() -> records.RunContext:
    return records.RunContext(
        challenger="glonet_1_degree",
        challenger_version="0.2.1",
        year=2024,
        region="global",
        oceanbench_version="0.2.1",
    )


def _frame(index_to_values: dict[str, list[float]], lead_days: list[int]) -> pandas.DataFrame:
    columns = [f"Lead day {lead}" for lead in lead_days]
    return pandas.DataFrame(list(index_to_values.values()), index=list(index_to_values.keys()), columns=columns)


def test_gridded_variable_records_carry_depth_and_standard_name(context):
    frame = _frame(
        {
            "Sea surface height (m) [sea_surface_height_above_geoid]{surface}": [0.07, 0.08],
            "Temperature (°C) [sea_water_potential_temperature]{50m}": [0.5, 0.6],
        },
        [1, 2],
    )
    result = records.gridded_rmsd_records(
        frame, reference="glorys", context=context, start_date="2024-01-03", depth_applicable=True
    )
    assert len(result) == 4
    surface = next(r for r in result if r["variable"] == "sea_surface_height_above_geoid" and r["lead_day"] == 1)
    assert surface["metric"] == "rmsd"
    assert surface["reference"] == "glorys"
    assert surface["depth"] == "surface"
    assert surface["unit"] == "m"
    assert surface["value"] == pytest.approx(0.07)
    assert surface["band"] is None and surface["polarity"] is None and surface["n"] is None
    fifty = next(r for r in result if r["depth"] == "50m")
    assert fifty["variable"] == "sea_water_potential_temperature"


def test_nan_values_become_null(context):
    frame = _frame({"Salinity (PSU) [sea_water_salinity]{100m}": [float("nan"), 0.1]}, [1, 2])
    result = records.gridded_rmsd_records(
        frame, reference="glo12", context=context, start_date="2024-01-03", depth_applicable=True
    )
    lead_one = next(r for r in result if r["lead_day"] == 1)
    assert lead_one["value"] is None
    lead_two = next(r for r in result if r["lead_day"] == 2)
    assert lead_two["value"] == pytest.approx(0.1)


def test_depth_agnostic_metrics_null_depth(context):
    frame = _frame({"Mixed layer depth (m) [ocean_mixed_layer_thickness]{surface}": [12.0]}, [1])
    result = records.gridded_rmsd_records(
        frame, reference="glorys", context=context, start_date="2024-01-03", depth_applicable=False
    )
    assert result[0]["depth"] is None
    assert result[0]["variable"] == "ocean_mixed_layer_thickness"


def test_lagrangian_records_start_at_lead_day_two(context):
    lead_days = [2, 3, 4, 5, 6, 7]
    frame = _frame(
        {"Lagrangian trajectory deviation (km) []{surface}": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]}, lead_days
    )
    result = records.lagrangian_records(frame, reference="glorys", context=context, start_date="2024-01-03")
    assert {r["lead_day"] for r in result} == set(lead_days)
    assert min(r["lead_day"] for r in result) == 2
    assert all(r["variable"] is None and r["depth"] is None for r in result)
    assert all(r["unit"] == "km" and r["metric"] == "lagrangian_deviation_km" for r in result)


def test_seven_day_horizon_is_respected(context):
    lead_days = list(range(1, 8))
    frame = _frame(
        {"Temperature (°C) [sea_water_potential_temperature]{surface}": [float(x) for x in lead_days]}, lead_days
    )
    result = records.gridded_rmsd_records(
        frame, reference="glorys", context=context, start_date="2024-01-03", depth_applicable=True
    )
    assert {r["lead_day"] for r in result} == set(range(1, 8))
    assert len(result) == 7


def test_class4_records_use_observations_reference_and_bins(context):
    frame = _frame(
        {
            "Sea level anomaly (m) [sea_surface_height_above_geoid]{surface}": [0.05, 0.06],
            "Temperature (°C) [sea_water_potential_temperature]{0-5m}": [0.4, 0.5],
            "Zonal current (m/s) [eastward_sea_water_velocity]{15m}": [0.1, 0.12],
        },
        [1, 2],
    )
    result = records.class4_records(frame, context=context, start_date=None)
    assert all(r["reference"] == "observations" and r["metric"] == "class4_rmsd" for r in result)
    assert all(r["start_date"] is None for r in result)
    depths = {r["depth"] for r in result}
    assert depths == {"surface", "0-5m", "15m"}


def test_records_to_dataframe_has_contract_schema(context):
    frame = _frame({"Sea surface height (m) [sea_surface_height_above_geoid]{surface}": [0.07, 0.08]}, [1, 2])
    result = records.gridded_rmsd_records(
        frame, reference="glorys", context=context, start_date="2024-01-03", depth_applicable=True
    )
    dataframe = records.records_to_dataframe(result)
    assert list(dataframe.columns) == records.SCORE_COLUMNS
    assert str(dataframe["year"].dtype) == "int32"
    assert str(dataframe["lead_day"].dtype) == "Int8"
    assert pandas.api.types.is_datetime64_any_dtype(dataframe["start_date"])


def test_empty_records_to_dataframe_keeps_columns():
    dataframe = records.records_to_dataframe([])
    assert list(dataframe.columns) == records.SCORE_COLUMNS
    assert len(dataframe) == 0


def test_clean_value_helper_handles_none_and_nan():
    assert records._clean_value(None) is None
    assert records._clean_value(float("nan")) is None
    assert records._clean_value(math.pi) == pytest.approx(math.pi)
