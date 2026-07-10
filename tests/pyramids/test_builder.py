# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pytest
import xarray

from oceanbench.core.attribution import COPERNICUS_MARINE_CREDIT
from oceanbench.core.schema_validation import validate_against_schema
from oceanbench.pyramids.builder import (
    LEAD_DAY_DIMENSION,
    START_DATE_DIMENSION,
    VariableSpec,
    build_pyramid,
)

try:
    import zarr
except ImportError:  # pragma: no cover
    zarr = None


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


@pytest.mark.skipif(zarr is None, reason="zarr required")
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


@pytest.mark.skipif(zarr is None, reason="zarr required")
def test_manifest_is_schema_valid(tmp_path):
    result = build_pyramid(
        _synthetic_layers(),
        _specs(),
        output_path=str(tmp_path / "synthetic.zarr"),
        tile_size=16,
    )
    validate_against_schema(result.manifest, "viewer-manifest")


@pytest.mark.skipif(zarr is None, reason="zarr required")
def test_manifest_carries_data_provenance(tmp_path, monkeypatch):
    from oceanbench.core.version import __version__ as oceanbench_version

    monkeypatch.setenv("OCEANBENCH_BUILD_COMMIT", "abc123def")
    result = build_pyramid(
        _synthetic_layers(),
        _specs(),
        output_path=str(tmp_path / "synthetic.zarr"),
        tile_size=16,
    )
    provenance = result.manifest["provenance"]
    assert provenance["oceanbench_version"] == oceanbench_version
    assert provenance["source_commit"] == "abc123def"
    assert provenance["generated_at"].endswith("Z")
    validate_against_schema(result.manifest, "viewer-manifest")


@pytest.mark.skipif(zarr is None, reason="zarr required")
def test_manifest_provenance_omits_source_commit_without_build_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OCEANBENCH_BUILD_COMMIT", raising=False)
    result = build_pyramid(
        _synthetic_layers(),
        _specs(),
        output_path=str(tmp_path / "synthetic.zarr"),
        tile_size=16,
    )
    assert "source_commit" not in result.manifest["provenance"]


@pytest.mark.skipif(zarr is None, reason="zarr required")
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


@pytest.mark.skipif(zarr is None, reason="zarr required")
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


@pytest.mark.skipif(zarr is None, reason="zarr required")
def test_root_attributes_carry_copernicus_attribution(tmp_path):
    result = build_pyramid(_synthetic_layers(), _specs(), output_path=str(tmp_path / "synthetic.zarr"), tile_size=16)
    root = zarr.open_group(result.zarr_path, mode="r")
    assert root.attrs["attribution"] == COPERNICUS_MARINE_CREDIT
    assert "disclaimer" in root.attrs


@pytest.mark.skipif(zarr is None, reason="zarr required")
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
