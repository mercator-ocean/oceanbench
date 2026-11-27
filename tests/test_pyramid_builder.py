# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import dataclasses
import logging

import numpy
import pytest
import xarray

from oceanbench.core.attribution import COPERNICUS_MARINE_CREDIT
from oceanbench.core.schema_validation import validate_against_schema
from oceanbench.pyramids import builder as builder_module
from oceanbench.pyramids import viewer_layers
from oceanbench.pyramids.builder import (
    LEAD_DAY_DIMENSION,
    START_DATE_DIMENSION,
    VariableSpec,
    build_pyramid,
)

import zarr


def _synthetic_layers(cell_size_deg: float = 0.25, span_deg: float = 8.0) -> xarray.Dataset:
    latitudes = numpy.arange(-span_deg, span_deg, cell_size_deg)
    longitudes = numpy.arange(-span_deg, span_deg, cell_size_deg)
    generator = numpy.random.default_rng(1234)
    temperature = generator.normal(15.0, 5.0, size=(2, 3, latitudes.size, longitudes.size))
    salinity = generator.normal(35.0, 1.0, size=(2, 3, latitudes.size, longitudes.size))
    temperature[..., :6, :6] = numpy.nan
    salinity[..., :6, :6] = numpy.nan
    dimensions = (START_DATE_DIMENSION, LEAD_DAY_DIMENSION, "latitude", "longitude")
    coordinates = {
        START_DATE_DIMENSION: numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]"),
        LEAD_DAY_DIMENSION: [1, 2, 3],
        "latitude": latitudes,
        "longitude": longitudes,
    }
    return xarray.Dataset(
        {
            "sea_water_potential_temperature": (dimensions, temperature),
            "sea_water_salinity": (dimensions, salinity),
        },
        coords=coordinates,
    ).chunk()


def _specs() -> dict[str, VariableSpec]:
    return {
        "sea_water_potential_temperature": VariableSpec(
            "sea_water_potential_temperature", "surface", "degC", "thermal"
        ),
        "sea_water_salinity": VariableSpec("sea_water_salinity", "surface", "PSU", "haline"),
    }


def test_multi_level_pyramid_tiling(tmp_path):
    result = build_pyramid(
        _synthetic_layers(),
        _specs(),
        output_path=str(tmp_path / "synthetic.zarr"),
        dataset_slug="synthetic",
        year=2024,
        tile_size=16,
    )
    assert result.level_count == 3
    assert [level["level"] for level in result.manifest["levels"]] == [0, 1, 2]
    root = zarr.open_group(result.zarr_path, mode="r")
    assert {"level"}.issubset(set(root.group_keys()))
    assert set(root["level"].group_keys()) == {"0", "1", "2"}


def test_manifest_is_schema_valid(tmp_path):
    result = build_pyramid(
        _synthetic_layers(),
        _specs(),
        output_path=str(tmp_path / "synthetic.zarr"),
        tile_size=16,
    )
    validate_against_schema(result.manifest, "viewer-manifest")


def test_manifest_carries_version_generation_time_and_column_flag(tmp_path):
    from oceanbench.core.version import __version__ as oceanbench_version

    result = build_pyramid(
        _synthetic_layers(),
        _specs(),
        output_path=str(tmp_path / "synthetic.zarr"),
        tile_size=16,
        has_columns=True,
    )
    assert result.manifest["provenance"]["oceanbench_version"] == oceanbench_version
    assert result.manifest["provenance"]["generated_at"].endswith("Z")
    assert result.manifest["has_columns"] is True
    validate_against_schema(result.manifest, "viewer-manifest")


def test_manifest_carries_the_optional_kind_and_description(tmp_path):
    result = build_pyramid(
        _synthetic_layers(),
        _specs(),
        output_path=str(tmp_path / "synthetic.zarr"),
        tile_size=16,
        kind="reference",
        description="GLORYS reanalysis",
    )
    assert result.manifest["kind"] == "reference"
    assert result.manifest["description"] == "GLORYS reanalysis"
    validate_against_schema(result.manifest, "viewer-manifest")
    with pytest.raises(Exception):
        validate_against_schema(result.manifest | {"kind": "oracle"}, "viewer-manifest")


def test_round_trip_error_within_quantization_step_and_land_preserved(tmp_path):
    layers = _synthetic_layers()
    result = build_pyramid(layers, _specs(), output_path=str(tmp_path / "synthetic.zarr"), tile_size=16)
    decoded = xarray.open_zarr(result.zarr_path, group="level/0")
    for variable_name in layers.data_vars:
        original = layers[variable_name].values
        decoded_values = decoded[variable_name].values
        step = result.manifest["variables"][variable_name]["scale_factor"]
        finite = numpy.isfinite(original)
        assert numpy.abs(decoded_values[finite] - original[finite]).max() <= step
        assert numpy.all(numpy.isnan(decoded_values[~finite]))


def test_land_mask_preserved_at_every_level(tmp_path):
    result = build_pyramid(_synthetic_layers(), _specs(), output_path=str(tmp_path / "synthetic.zarr"), tile_size=16)
    for level in result.manifest["levels"]:
        decoded = xarray.open_zarr(result.zarr_path, group=f"level/{level['level']}")
        corner = (
            decoded["sea_water_potential_temperature"]
            .isel({START_DATE_DIMENSION: 0, LEAD_DAY_DIMENSION: 0})
            .values[0, 0]
        )
        assert numpy.isnan(corner)


def test_root_attributes_carry_copernicus_attribution(tmp_path):
    result = build_pyramid(_synthetic_layers(), _specs(), output_path=str(tmp_path / "synthetic.zarr"), tile_size=16)
    root = zarr.open_group(result.zarr_path, mode="r")
    assert root.attrs["attribution"] == COPERNICUS_MARINE_CREDIT
    assert "disclaimer" in root.attrs


def test_single_level_for_one_degree_layers(tmp_path):
    result = build_pyramid(
        _synthetic_layers(cell_size_deg=1.0, span_deg=20.0),
        _specs(),
        output_path=str(tmp_path / "one_degree.zarr"),
        tile_size=256,
    )
    assert result.level_count == 1
    assert [level["level"] for level in result.manifest["levels"]] == [0]


def test_missing_variable_spec_is_rejected(tmp_path):
    layers = _synthetic_layers()
    with pytest.raises(ValueError):
        build_pyramid(layers, {}, output_path=str(tmp_path / "synthetic.zarr"))


def _temperature_with_display_range(maximum_value: float) -> tuple[xarray.Dataset, dict[str, VariableSpec]]:
    layers = _synthetic_layers()[["sea_water_potential_temperature"]].load()
    layers["sea_water_potential_temperature"][0, 0, 10, 10] = maximum_value
    specs = {
        "sea_water_potential_temperature": VariableSpec(
            "sea_water_potential_temperature", "surface", "degC", "thermal", default_range=(-2.0, 32.0)
        )
    }
    return layers, specs


def test_a_spurious_extreme_value_that_blurs_the_display_range_is_rejected(tmp_path):
    layers, specs = _temperature_with_display_range(1.0e4)

    with pytest.raises(ValueError, match="sea_water_potential_temperature.*10000.*quantization step"):
        build_pyramid(layers, specs, output_path=str(tmp_path / "spurious.zarr"), tile_size=16)


def test_a_field_resolving_its_display_range_records_its_quantization_step(tmp_path):
    layers, specs = _temperature_with_display_range(40.0)

    result = build_pyramid(layers, specs, output_path=str(tmp_path / "sane.zarr"), tile_size=16)

    entry = result.manifest["variables"]["sea_water_potential_temperature"]
    assert entry["quantization_step"] == entry["scale_factor"]
    assert entry["quantization_step"] <= 34.0 / 2000


def _salinity_with_outliers() -> xarray.Dataset:
    layers = _synthetic_layers()[["sea_water_salinity"]].load()
    layers["sea_water_salinity"][0, 0, 10, 10] = -0.0003
    layers["sea_water_salinity"][1, 2, 20, 20] = 42.1
    return layers.chunk({START_DATE_DIMENSION: 1, LEAD_DAY_DIMENSION: 1})


def test_the_default_range_ignores_outliers_and_the_data_range_keeps_them(tmp_path):
    layers = _salinity_with_outliers()

    result = build_pyramid(layers, _specs(), output_path=str(tmp_path / "salinity.zarr"), tile_size=16)

    entry = result.manifest["variables"]["sea_water_salinity"]
    values = layers["sea_water_salinity"].values
    expected = numpy.nanpercentile(values, [2.0, 98.0])
    assert entry["default_range"] == pytest.approx(expected, abs=0.05)
    assert entry["data_range"] == [-0.0003, 42.1]
    # The quantization still spans the true minimum and maximum.
    assert entry["add_offset"] < -0.0003
    assert entry["add_offset"] + 65534 * entry["scale_factor"] > 42.1
    validate_against_schema(result.manifest, "viewer-manifest")


def test_the_default_range_is_computed_on_a_regular_subsample(tmp_path, monkeypatch):
    monkeypatch.setattr(builder_module, "ROBUST_RANGE_SAMPLE_SIZE", 1600)
    layers = _salinity_with_outliers()
    assert builder_module.robust_range_stride(layers) == 4

    result = build_pyramid(layers, _specs(), output_path=str(tmp_path / "salinity.zarr"), tile_size=16)

    subsample = layers["sea_water_salinity"].values[..., ::4, ::4]
    expected = numpy.nanpercentile(subsample, [2.0, 98.0])
    assert result.manifest["variables"]["sea_water_salinity"]["default_range"] == pytest.approx(expected)


def test_a_declared_default_range_is_kept(tmp_path):
    layers, specs = _temperature_with_display_range(40.0)

    result = build_pyramid(layers, specs, output_path=str(tmp_path / "declared.zarr"), tile_size=16)

    entry = result.manifest["variables"]["sea_water_potential_temperature"]
    assert entry["default_range"] == [-2.0, 32.0]
    assert entry["data_range"][1] == 40.0


def test_an_extreme_value_against_a_robust_range_only_warns(tmp_path, caplog):
    layers, specs = _temperature_with_display_range(1.0e4)
    undeclared = {name: dataclasses.replace(spec, default_range=None) for name, spec in specs.items()}

    with caplog.at_level(logging.WARNING, logger="oceanbench.pyramids.builder"):
        result = build_pyramid(layers, undeclared, output_path=str(tmp_path / "spurious.zarr"), tile_size=16)

    assert result.manifest["variables"]["sea_water_potential_temperature"]["data_range"][1] == 1.0e4
    assert any("quantization step" in record.getMessage() for record in caplog.records)


def test_layer_units_in_the_zarr_attributes_match_the_manifest(tmp_path):
    challenger = xarray.Dataset(
        {
            "sea_water_potential_temperature": (
                ("first_day_datetime", "lead_day_index", "depth", "latitude", "longitude"),
                numpy.full((1, 2, 2, 4, 4), 15.0),
                {"units": "degrees_C"},
            ),
            "sea_water_salinity": (
                ("first_day_datetime", "lead_day_index", "depth", "latitude", "longitude"),
                numpy.full((1, 2, 2, 4, 4), 35.0),
                {"units": "1e-3"},
            ),
        },
        coords={
            "first_day_datetime": numpy.array(["2024-01-03"], dtype="datetime64[ns]"),
            "lead_day_index": [0, 1],
            "depth": [0.5, 20.0],
            "latitude": numpy.arange(4.0),
            "longitude": numpy.arange(4.0),
        },
    )
    layers, specs = viewer_layers(challenger)
    result = build_pyramid(layers, specs, output_path=str(tmp_path / "units.zarr"), tile_size=16)

    level = zarr.open_group(result.zarr_path, mode="r")["level/0"]
    for name, entry in result.manifest["variables"].items():
        assert level[name].attrs["units"] == entry["units"]
    assert result.manifest["variables"]["sea_water_potential_temperature"]["units"] == "°C"
    assert result.manifest["variables"]["sea_water_salinity"]["units"] == "PSU"
