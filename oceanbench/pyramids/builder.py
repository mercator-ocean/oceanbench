# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass
import datetime
import json
from pathlib import Path

import dask
import numcodecs
import numpy
import xarray
import zarr

from oceanbench.core.attribution import copernicus_marine_attribution_attrs
from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.schema_validation import validate_against_schema
from oceanbench.core.version import __version__ as OCEANBENCH_VERSION
from oceanbench.pyramids import levels as level_planning
from oceanbench.pyramids.quantization import Quantization, quantization_for_range, zarr_encoding

START_DATE_DIMENSION = "start_date"
LEAD_DAY_DIMENSION = "lead_day"
# A native 1/12 degree tile of 1024 cells is about 1 MB compressed, one browser-friendly fetch,
# and the coarse levels collapse to a single tile. The pyramid is zarr v2 since zarr 2 cannot write shards.
DEFAULT_TILE_SIZE = 1024
_DEFLATE_COMPRESSION_LEVEL = 6


@dataclass(frozen=True)
class VariableSpec:
    """Display and provenance metadata for one viewer layer (encoding is derived)."""

    standard_name: str
    depth: str
    units: str
    default_colormap: str
    default_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class PyramidResult:
    zarr_path: str
    manifest_path: str
    manifest: dict
    level_count: int


def _tile_chunks(latitude_size: int, longitude_size: int, tile_size: int) -> dict[str, int]:
    return {
        START_DATE_DIMENSION: 1,
        LEAD_DAY_DIMENSION: 1,
        Dimension.LATITUDE.key(): min(tile_size, latitude_size),
        Dimension.LONGITUDE.key(): min(tile_size, longitude_size),
    }


def _variable_data_ranges(layers: xarray.Dataset) -> dict[str, tuple[float, float]]:
    minimums, maximums = dask.compute(layers.min(skipna=True), layers.max(skipna=True))
    return {name: (float(minimums[name].values), float(maximums[name].values)) for name in layers.data_vars}


def deflate_compressor() -> numcodecs.abc.Codec:
    """Zlib, which browsers decode natively with ``DecompressionStream('deflate')``, so the viewer needs no codec."""
    return numcodecs.Zlib(level=_DEFLATE_COMPRESSION_LEVEL)


def _write_level(
    layers: xarray.Dataset,
    quantizations: dict[str, Quantization],
    *,
    store_path: Path,
    level_index: int,
    tile_size: int,
    is_first_level: bool,
) -> None:
    latitude_size = int(layers.sizes[Dimension.LATITUDE.key()])
    longitude_size = int(layers.sizes[Dimension.LONGITUDE.key()])
    chunks = _tile_chunks(latitude_size, longitude_size, tile_size)
    tiled_layers = layers.chunk(chunks)
    for variable_name in tiled_layers.variables:
        tiled_layers[variable_name].encoding = {}
    compressor = deflate_compressor()
    encoding = {
        variable_name: zarr_encoding(quantizations[variable_name], compressor)
        for variable_name in tiled_layers.data_vars
    }
    coordinate_names = set(tiled_layers.variables) - set(tiled_layers.data_vars)
    for coordinate_name in coordinate_names:
        encoding[coordinate_name] = {"compressor": compressor}
    tiled_layers.to_zarr(
        store_path,
        group=f"level/{level_index}",
        mode="w" if is_first_level else "a",
        encoding=encoding,
        consolidated=False,
    )


def _variable_manifest_entry(spec: VariableSpec, quantization: Quantization, data_range: tuple[float, float]) -> dict:
    default_range = list(spec.default_range) if spec.default_range is not None else [data_range[0], data_range[1]]
    return {
        "standard_name": spec.standard_name,
        "depth": spec.depth,
        "units": spec.units,
        "scale_factor": quantization.scale_factor,
        "add_offset": quantization.add_offset,
        "fill_value": quantization.fill_value,
        "default_colormap": spec.default_colormap,
        "default_range": [float(default_range[0]), float(default_range[1])],
    }


def _bounds(layers: xarray.Dataset) -> dict:
    latitudes = layers[Dimension.LATITUDE.key()].values
    longitudes = layers[Dimension.LONGITUDE.key()].values
    return {
        "minimum_latitude": float(numpy.min(latitudes)),
        "maximum_latitude": float(numpy.max(latitudes)),
        "minimum_longitude": float(numpy.min(longitudes)),
        "maximum_longitude": float(numpy.max(longitudes)),
    }


def _manifest_provenance() -> dict:
    # The viewer uses generated_at as the cache token of the dataset stores.
    return {
        "oceanbench_version": OCEANBENCH_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _manifest(
    layers: xarray.Dataset,
    specs: dict[str, VariableSpec],
    quantizations: dict[str, Quantization],
    data_ranges: dict[str, tuple[float, float]],
    level_plans: list[level_planning.LevelPlan],
    *,
    dataset_slug: str | None,
    year: int | None,
    tile_size: int,
    has_columns: bool,
) -> dict:
    start_dates = [numpy.datetime_as_string(value, unit="D") for value in layers[START_DATE_DIMENSION].values]
    lead_days = [int(value) for value in layers[LEAD_DAY_DIMENSION].values]
    manifest = {
        "levels": [
            {
                "level": plan.level,
                "cell_size_deg": plan.cell_size_deg,
                "latitude_size": plan.latitude_size,
                "longitude_size": plan.longitude_size,
            }
            for plan in level_plans
        ],
        "tile_size": tile_size,
        "bounds": _bounds(layers),
        "variables": {
            name: _variable_manifest_entry(specs[name], quantizations[name], data_ranges[name])
            for name in layers.data_vars
        },
        "start_dates": start_dates,
        "lead_days": lead_days,
        "has_columns": has_columns,
        "provenance": _manifest_provenance(),
    }
    if dataset_slug is not None:
        manifest["dataset"] = dataset_slug
    if year is not None:
        manifest["year"] = year
    return manifest


def _write_root_attributes(store_path: Path, dataset_slug: str | None, year: int | None, tile_size: int) -> None:
    root_group = zarr.open_group(str(store_path), mode="a")
    root_group.attrs.update(copernicus_marine_attribution_attrs())
    root_group.attrs["tile_size"] = tile_size
    if dataset_slug is not None:
        root_group.attrs["dataset"] = dataset_slug
    if year is not None:
        root_group.attrs["year"] = year


def build_pyramid(
    layers: xarray.Dataset,
    specs: dict[str, VariableSpec],
    *,
    output_path: str,
    dataset_slug: str | None = None,
    year: int | None = None,
    tile_size: int = DEFAULT_TILE_SIZE,
    has_columns: bool = False,
) -> PyramidResult:
    """
    Build the pyramid zarr and its ``<slug>.viewer-manifest.json`` beside it.

    ``layers`` has dims ``(start_date, lead_day, latitude, longitude)`` and one float variable per
    viewer layer, stored as quantized uint16 tiles in groups ``level/<k>`` from the native grid up to
    about one degree. ``has_columns`` tells the viewer whether a water-column store sits beside it.
    """
    missing_specs = set(layers.data_vars) - set(specs)
    if missing_specs:
        raise ValueError(f"Missing variable specs for: {sorted(missing_specs)}")
    store_path = Path(output_path)
    manifest_path = store_path.with_name(store_path.name.replace(".zarr", "") + ".viewer-manifest.json")

    data_ranges = _variable_data_ranges(layers)
    quantizations = {
        name: quantization_for_range(data_ranges[name][0], data_ranges[name][1]) for name in layers.data_vars
    }

    level_plans = level_planning.plan_levels(layers)
    current_layers = layers
    for plan in level_plans:
        _write_level(
            current_layers,
            quantizations,
            store_path=store_path,
            level_index=plan.level,
            tile_size=tile_size,
            is_first_level=(plan.level == 0),
        )
        if plan.level != level_plans[-1].level:
            current_layers = level_planning.coarsen_by_two(current_layers)

    _write_root_attributes(store_path, dataset_slug, year, tile_size)
    zarr.consolidate_metadata(str(store_path))

    manifest = _manifest(
        layers,
        specs,
        quantizations,
        data_ranges,
        level_plans,
        dataset_slug=dataset_slug,
        year=year,
        tile_size=tile_size,
        has_columns=has_columns,
    )
    validate_against_schema(manifest, "viewer-manifest")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")

    return PyramidResult(
        zarr_path=str(store_path),
        manifest_path=str(manifest_path),
        manifest=manifest,
        level_count=len(level_plans),
    )
