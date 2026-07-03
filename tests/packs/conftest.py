# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Synthetic, network-free fixtures for the evaluation-pack tests.

Builds a tiny 1-degree forecast, a surface reference pack and a "published" scores parquet on
disk so the local-evaluation path (pack manifest resolution, runner reuse, aggregation and the
overlay scorecard) can be exercised end to end without the staged reference data or the network.
"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

import numpy
import pytest
import xarray

from oceanbench.core.dataset_utils import DepthLevel
from oceanbench.packs.manifest import write_pack_manifest
from oceanbench.runner.run import run_challenger_scores

_LATITUDES = numpy.arange(20.5, 26.5, 1.0)
_LONGITUDES = numpy.arange(0.5, 6.5, 1.0)
_DEPTHS = numpy.array([DepthLevel.SURFACE.value, 47.37369, 92.32607], dtype="float64")
_START_DATES = numpy.array(
    [
        numpy.datetime64(datetime(2024, 1, 3)),
        numpy.datetime64(datetime(2024, 1, 10)),
        numpy.datetime64(datetime(2024, 1, 17)),
    ]
)
_LEAD_DAY_INDICES = numpy.arange(4)

_VOLUME_VARIABLES = [
    "sea_water_potential_temperature",
    "sea_water_salinity",
    "eastward_sea_water_velocity",
    "northward_sea_water_velocity",
]
_SURFACE_VARIABLE = "sea_surface_height_above_geoid"


def _synthetic_dataset(seed: int, offset: float = 0.0) -> xarray.Dataset:
    generator = numpy.random.default_rng(seed)
    volume_shape = (len(_START_DATES), len(_LEAD_DAY_INDICES), len(_DEPTHS), len(_LATITUDES), len(_LONGITUDES))
    surface_shape = (len(_START_DATES), len(_LEAD_DAY_INDICES), len(_LATITUDES), len(_LONGITUDES))
    volume_dimensions = ("first_day_datetime", "lead_day_index", "depth", "latitude", "longitude")
    surface_dimensions = ("first_day_datetime", "lead_day_index", "latitude", "longitude")
    data_variables = {
        name: (volume_dimensions, offset + generator.standard_normal(volume_shape)) for name in _VOLUME_VARIABLES
    }
    data_variables[_SURFACE_VARIABLE] = (surface_dimensions, offset + generator.standard_normal(surface_shape))
    return xarray.Dataset(
        data_variables,
        coords={
            "first_day_datetime": _START_DATES,
            "lead_day_index": _LEAD_DAY_INDICES,
            "depth": _DEPTHS,
            "latitude": _LATITUDES,
            "longitude": _LONGITUDES,
        },
    )


def _write_zarr(dataset: xarray.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(str(path), mode="w", consolidated=True)


@dataclass(frozen=True)
class LocalEvaluationFixture:
    forecast_path: str
    pack_directory: str
    published_scores_path: str
    published_challengers_path: str
    published_challenger_slug: str


@pytest.fixture
def local_evaluation_fixture(tmp_path: Path) -> LocalEvaluationFixture:
    forecast = _synthetic_dataset(seed=1)
    reference = _synthetic_dataset(seed=2, offset=0.3)
    surface_reference = reference.sel(depth=[DepthLevel.SURFACE.value], method="nearest")

    forecast_path = tmp_path / "forecast.zarr"
    _write_zarr(forecast, forecast_path)

    pack_directory = tmp_path / "pack"
    _write_zarr(surface_reference, pack_directory / "references" / "glorys.zarr")

    observation_stub = xarray.Dataset(
        {_SURFACE_VARIABLE: (("observations",), numpy.zeros(len(_START_DATES)))},
        coords={
            "first_day_datetime": (("observations",), _START_DATES),
            "time": (("observations",), _START_DATES),
        },
    )
    _write_zarr(observation_stub, pack_directory / "observations" / "observations.zarr")

    manifest = {
        "schema_version": "1.0",
        "kind": "quick",
        "year": 2024,
        "region": "global",
        "resolution": "one_degree",
        "oceanbench_version": "0.0.0-test",
        "generated_at": "2026-07-04T00:00:00+00:00",
        "start_dates": ["2024-01-03", "2024-01-10", "2024-01-17"],
        "attribution": "Generated using E.U. Copernicus Marine Service Information; test",
        "disclaimer": "test disclaimer",
        "source_products": {"GLORYS12": "GLOBAL_MULTIYEAR_PHY_001_030"},
        "upstream": [{"name": "glorys", "product_id": "GLOBAL_MULTIYEAR_PHY_001_030", "retrieved": "2026-07-04"}],
        "contents": {
            "references": {
                "glorys": {
                    "path": "references/glorys.zarr",
                    "variables": [_SURFACE_VARIABLE],
                    "depths": ["surface"],
                }
            },
            "observations": {"path": "observations/observations.zarr"},
            "baselines": {},
        },
        "baselines_available": False,
        "notes": ["Baselines (climatology / persistence) are not available yet."],
    }
    write_pack_manifest(manifest, str(pack_directory))

    published_slug = "glonet_1_degree"
    published_run = run_challenger_scores(
        published_slug,
        "global",
        2024,
        references=("glorys",),
        include_gridded=True,
        include_mixed_layer_depth=False,
        include_geostrophic=True,
        include_class4=False,
        include_lagrangian=False,
        area_weighted=True,
        output_root=str(tmp_path / "published_run"),
        dataset=forecast,
        reference_openers={"glorys": lambda challenger: reference},
    )
    published_scores_path = tmp_path / "published.parquet"
    published_run.scores.to_parquet(str(published_scores_path), index=False)

    published_challengers_path = tmp_path / "challengers.json"
    published_challengers_path.write_text(
        json.dumps({published_slug: {"display_name": "GLONET (1°)", "is_baseline": False}}),
        encoding="utf-8",
    )

    return LocalEvaluationFixture(
        forecast_path=str(forecast_path),
        pack_directory=str(pack_directory),
        published_scores_path=str(published_scores_path),
        published_challengers_path=str(published_challengers_path),
        published_challenger_slug=published_slug,
    )
