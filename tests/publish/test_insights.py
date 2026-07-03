# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path

import pytest

from oceanbench.core.schema_validation import validate_against_schema
from oceanbench.publish import insights


def _spectrum_entry(reference: str) -> dict:
    return {
        "variable": "sea_surface_height_above_geoid",
        "region": "gulfstream",
        "lead_day": 1,
        "reference": reference,
        "unit": "m^2",
        "wavelength": [4000.0, 2000.0, 1000.0, 500.0],
        "challenger_power": [1.0, 0.8, 0.4, None],
        "reference_power": [1.0, 0.9, 0.7, 0.5],
        "error_power": [0.0, 0.1, 0.3, None],
    }


def _eddy(identifier: int, polarity: str) -> dict:
    return {
        "id": identifier,
        "latitude": 38.5,
        "longitude": -60.25,
        "polarity": polarity,
        "contour_latitude": [38.0, 38.5, 39.0],
        "contour_longitude": [-61.0, -60.0, -60.5],
    }


def _eddy_census(reference: str) -> dict:
    return {
        "reference": reference,
        "frames": [
            {
                "lead_day": 1,
                "matches": [
                    {
                        "challenger": _eddy(0, "anticyclone"),
                        "reference": _eddy(3, "anticyclone"),
                        "displacement_km": 42.5,
                    }
                ],
                "spurious": [_eddy(1, "cyclone")],
                "missed": [_eddy(4, "cyclone")],
            }
        ],
    }


def test_spectra_payload_validates_against_schema() -> None:
    payload = insights.build_spectra_payload([_spectrum_entry("glorys"), _spectrum_entry("glo12")])
    validate_against_schema(payload, "spectra")
    assert payload["kind"] == "spectra"
    assert len(payload["entries"]) == 2


def test_eddies_payload_validates_against_schema() -> None:
    payload = insights.build_eddies_payload(
        [_eddy_census("glorys")],
        variable="sea_surface_height_above_geoid",
        bounds={
            "minimum_latitude": 30.0,
            "maximum_latitude": 45.0,
            "minimum_longitude": -80.0,
            "maximum_longitude": -50.0,
        },
    )
    validate_against_schema(payload, "eddies")
    assert payload["references"][0]["reference"] == "glorys"


def test_write_realism_insights_round_trips_and_registers_in_manifest(tmp_path: Path) -> None:
    output_directory = tmp_path / "insights"
    result = insights.write_realism_insights(
        [_spectrum_entry("glorys")],
        [_eddy_census("glorys")],
        str(output_directory),
        variable="sea_surface_height_above_geoid",
    )

    manifest = result.manifest_result.manifest
    assert set(manifest) == {"spectra", "eddies"}
    assert manifest["spectra"]["kind"] == "spectra"
    assert manifest["eddies"]["kind"] == "eddies"

    for semantic_key, schema_name in (("spectra", "spectra"), ("eddies", "eddies")):
        blob_path = output_directory / manifest[semantic_key]["url"]
        assert blob_path.exists()
        assert blob_path.stat().st_size == manifest[semantic_key]["bytes"]
        round_tripped_payload = insights.read_insight_payload(str(blob_path))
        validate_against_schema(round_tripped_payload, schema_name)

    assert insights.read_insight_payload(str(output_directory / manifest["spectra"]["url"])) == result.spectra_payload


def test_invalid_spectrum_entry_is_rejected() -> None:
    broken_entry = _spectrum_entry("glorys")
    broken_entry["lead_day"] = 0  # violates minimum 1 in the schema
    with pytest.raises(Exception):
        insights.build_spectra_payload([broken_entry])
