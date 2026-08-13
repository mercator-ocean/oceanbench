# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Class-4 overlay extracts: round-trip, honest counts, decimation, size budget, manifest."""

import json

import numpy
import pandas
import pytest

from oceanbench.publish.class4_overlays import (
    DISPLAY_POINT_CAP,
    OVERLAY_FILE_SUFFIX,
    OVERLAY_MAGIC,
    OVERLAY_VALUE_COLUMNS,
    decode_class4_overlay,
    encode_class4_overlay,
    overlay_index_entry,
    overlay_relative_path,
    read_class4_overlay,
    read_class4_overlay_manifest,
    write_class4_overlay_manifest,
    write_class4_overlays,
)

SELECTION = {
    "dataset_slug": "glonet",
    "region": "global",
    "variable": "sea_water_potential_temperature",
    "depth_bin": "100-300m",
    "start_date": "2024-06-15",
    "lead_day": 3,
}


def _matchup_frame(point_count: int, *, seed: int = 0, starts=("2024-06-15",), leads=(1, 3)) -> pandas.DataFrame:
    generator = numpy.random.default_rng(seed)
    frames = []
    for start_date in starts:
        for lead_day in leads:
            for variable, depth_bin in (
                ("sea_surface_height_above_geoid", "surface"),
                ("sea_water_potential_temperature", "100-300m"),
            ):
                frames.append(
                    pandas.DataFrame(
                        {
                            "variable": variable,
                            "depth_bin": depth_bin,
                            "lead_day": numpy.int16(lead_day),
                            "start_date": start_date,
                            "latitude": generator.uniform(-80, 85, point_count),
                            "longitude": generator.uniform(-180, 180, point_count),
                            "observation_value": generator.uniform(0, 30, point_count),
                            "model_value": generator.uniform(0, 30, point_count),
                        }
                    )
                )
    return pandas.concat(frames, ignore_index=True)


def test_round_trip_recovers_values_within_the_quantization_step():
    generator = numpy.random.default_rng(1)
    latitude = generator.uniform(-80, 85, 5000)
    longitude = generator.uniform(-180, 180, 5000)
    observation = generator.uniform(-2, 32, 5000)
    model = observation + generator.normal(0, 0.5, 5000)

    payload, entry = encode_class4_overlay(latitude, longitude, observation, model, **SELECTION)
    header, decoded = decode_class4_overlay(payload)

    assert payload[:4] == OVERLAY_MAGIC
    assert list(decoded) == list(OVERLAY_VALUE_COLUMNS)
    assert entry.displayed_count == 5000
    for name, expected in zip(OVERLAY_VALUE_COLUMNS, (latitude, longitude, observation, model)):
        step = header["quantization"][name]["scale_factor"]
        assert numpy.abs(decoded[name] - expected).max() <= step
        assert decoded[name].dtype == numpy.float64


def test_header_carries_the_selection_key_and_no_decimation_under_the_cap():
    payload, entry = encode_class4_overlay(
        numpy.zeros(10), numpy.zeros(10), numpy.ones(10), numpy.ones(10) + 0.5, **SELECTION
    )
    header, _ = decode_class4_overlay(payload)

    assert header["variable"] == SELECTION["variable"]
    assert header["depth_bin"] == SELECTION["depth_bin"]
    assert header["start_date"] == SELECTION["start_date"]
    assert header["lead_day"] == 3
    assert header["decimated"] is False
    assert (header["observation_count"], header["matched_count"], header["displayed_count"]) == (10, 10, 10)
    assert entry.relative_path.endswith(OVERLAY_FILE_SUFFIX)


def test_non_finite_pairs_are_dropped_and_counted_not_hidden():
    latitude = numpy.arange(6, dtype=float)
    observation = numpy.array([1.0, numpy.nan, 3.0, 4.0, 5.0, 6.0])
    model = numpy.array([1.0, 2.0, 3.0, numpy.nan, 5.0, 6.0])

    payload, entry = encode_class4_overlay(latitude, latitude, observation, model, **SELECTION)
    header, decoded = decode_class4_overlay(payload)

    assert entry.observation_count == 6
    assert entry.matched_count == 4
    assert entry.displayed_count == 4
    assert header["decimated"] is False
    assert decoded["latitude"].size == 4


def test_decimation_caps_the_file_reports_the_full_count_and_is_reproducible():
    generator = numpy.random.default_rng(2)
    count = 40_000
    values = generator.uniform(0, 30, count)
    first, entry = encode_class4_overlay(values, values, values, values, display_point_cap=1000, **SELECTION)
    second, _ = encode_class4_overlay(values, values, values, values, display_point_cap=1000, **SELECTION)
    header, decoded = decode_class4_overlay(first)

    assert first == second
    assert header["decimated"] is True
    assert header["observation_count"] == count
    assert header["matched_count"] == count
    assert header["displayed_count"] == 1000
    assert decoded["latitude"].size == 1000
    assert entry.byte_size < 1000 * 8 + 2048


def test_a_capped_extract_stays_inside_the_first_paint_budget():
    generator = numpy.random.default_rng(3)
    count = 400_000
    values = generator.uniform(-2, 32, count)
    payload, entry = encode_class4_overlay(values, values, values, values, **SELECTION)

    assert entry.displayed_count == DISPLAY_POINT_CAP
    assert entry.byte_size == len(payload)
    assert len(payload) < 512 * 1024


def test_write_class4_overlays_writes_one_file_per_selection_at_the_contract_path(tmp_path):
    frame = _matchup_frame(50, starts=("2024-06-15", "2024-06-22"), leads=(1, 3))
    entries = write_class4_overlays(frame, str(tmp_path), dataset_slug="glonet", region="global")

    assert len(entries) == 8
    for entry in entries:
        assert (tmp_path / entry.relative_path).is_file()
        assert entry.relative_path == overlay_relative_path(
            entry.variable, entry.depth_bin, entry.start_date, entry.lead_day
        )
    assert (tmp_path / "sea_water_potential_temperature" / "100-300m" / "2024-06-15-lead03.obx").is_file()

    header, decoded = read_class4_overlay(
        str(tmp_path / "sea_water_potential_temperature" / "100-300m" / "2024-06-15-lead03.obx")
    )
    expected = frame[
        (frame["variable"] == "sea_water_potential_temperature")
        & (frame["start_date"] == "2024-06-15")
        & (frame["lead_day"] == 3)
    ]
    assert header["displayed_count"] == len(expected)
    step = header["quantization"]["observation_value"]["scale_factor"]
    assert numpy.abs(numpy.sort(decoded["observation_value"]) - numpy.sort(expected["observation_value"])).max() <= step


def test_write_class4_overlays_rejects_a_frame_missing_overlay_columns(tmp_path):
    frame = _matchup_frame(5).drop(columns=["model_value"])
    with pytest.raises(ValueError, match="model_value"):
        write_class4_overlays(frame, str(tmp_path), dataset_slug="glonet", region="global")


def test_manifest_indexes_every_extract_with_its_counts_and_the_url_template(tmp_path):
    frame = _matchup_frame(50, starts=("2024-06-15", "2024-06-22"), leads=(1, 3))
    entries = write_class4_overlays(frame, str(tmp_path), dataset_slug="glonet", region="global")
    manifest_path = write_class4_overlay_manifest(entries, str(tmp_path), dataset_slug="glonet", region="global")
    manifest = read_class4_overlay_manifest(manifest_path)

    assert manifest["extract_count"] == 8
    assert manifest["format"] == "obx/1"
    assert manifest["counts"] == ["observation_count", "matched_count", "displayed_count"]
    availability = manifest["availability"]["sea_water_potential_temperature"]["100-300m"]["2024-06-15"]
    assert availability["3"] == [50, 50, 50]
    assert manifest["template"].endswith("{start_date}-lead{lead_day}.obx")
    assert manifest["total_byte_size"] == sum(entry.byte_size for entry in entries)


def test_index_entry_points_at_the_published_layout():
    entry = overlay_index_entry("glonet", "ibi")

    assert entry["template"] == (
        "./data/insights/glonet/ibi/class4-overlays/{variable}/{depth_bin}/{start_date}-lead{lead_day}.obx"
    )
    assert entry["manifest"] == "./data/insights/glonet/ibi/class4-overlays/manifest.json"
    assert entry["display_point_cap"] == DISPLAY_POINT_CAP
    assert json.dumps(entry)


def test_a_path_component_can_never_escape_the_overlay_directory():
    with pytest.raises(ValueError):
        overlay_relative_path("../secrets", "surface", "2024-06-15", 1)
