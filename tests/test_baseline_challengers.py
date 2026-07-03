# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Baseline challengers (climatology, persistence): SLA shift, depths, registry, wiring.

These cover the port from branch ``291-add-persistence-climatology``: the per-challenger
SSH-to-SLA shift keyed on ``oceanbench_source_name`` (climatology -0.1329, everyone else the
GLO12-calibrated default -0.1148), the 15 m current depth bracket in the baseline-generation
scored depths, and the in-repo ``challengers.json`` registry consumed by the score page.
"""

import ast
import json
import os
from pathlib import Path

import numpy
import pytest
import xarray

import oceanbench.core.classIV_support as classIV_support
import oceanbench.core.challenger_datasets as core_challenger_datasets
import oceanbench.datasets.challenger as challenger_datasets
import oceanbench.runner.run as runner_run
from oceanbench.core.classIV_support import (
    DEFAULT_MEAN_SEA_SURFACE_HEIGHT_SHIFT,
    mean_sea_surface_height_shift,
    prepare_class4_model_variable,
)
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.schema_validation import validate_against_schema
from oceanbench.publish import publish_challengers_registry

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_BASELINE_SLUGS = ["persistence", "persistence_1_degree", "climatology", "climatology_1_degree"]
_OFFICIAL_SLUGS = [
    "glo12",
    "glo12_1_degree",
    "glonet",
    "glonet_1_degree",
    "wenhai",
    "wenhai_1_degree",
    "xihe",
    "xihe_1_degree",
    "langya",
    "langya_1_degree",
]


def _ssh_model_variable() -> xarray.DataArray:
    return xarray.DataArray(
        numpy.array([[1.0, 1.0], [1.0, 1.0]]),
        dims=[Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={
            Dimension.LATITUDE.key(): [0.0, 1.0],
            Dimension.LONGITUDE.key(): [10.0, 11.0],
        },
        name=Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
    )


@pytest.mark.parametrize(
    "challenger_name, expected_shift",
    [
        (None, -0.1148),
        ("glo12", -0.1148),
        ("persistence", -0.1148),
        ("climatology", -0.1329),
    ],
)
def test_ssh_to_sla_uses_per_challenger_mean_sea_surface_height_shift(
    monkeypatch, challenger_name, expected_shift
) -> None:
    monkeypatch.setattr(classIV_support, "get_dataset_resolution", lambda dataset: "native")
    monkeypatch.setattr(
        classIV_support,
        "load_mean_dynamic_topography",
        lambda resolution: xarray.DataArray(0.0),
    )

    model_variable = _ssh_model_variable()
    sea_level_anomaly = prepare_class4_model_variable(
        model_variable,
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
        challenger_name,
    )

    numpy.testing.assert_allclose(sea_level_anomaly.values, model_variable.values - expected_shift)


def test_mean_sea_surface_height_shift_lookup() -> None:
    assert mean_sea_surface_height_shift("climatology") == -0.1329
    assert mean_sea_surface_height_shift("persistence") == DEFAULT_MEAN_SEA_SURFACE_HEIGHT_SHIFT
    assert mean_sea_surface_height_shift("glo12") == DEFAULT_MEAN_SEA_SURFACE_HEIGHT_SHIFT
    assert mean_sea_surface_height_shift(None) == DEFAULT_MEAN_SEA_SURFACE_HEIGHT_SHIFT
    assert DEFAULT_MEAN_SEA_SURFACE_HEIGHT_SHIFT == -0.1148


def _scored_depths_from_generation_script() -> list[float]:
    script_path = _REPOSITORY_ROOT / "helper_scripts" / "baseline_generation" / "compute_glorys_climatology.py"
    module = ast.parse(script_path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SCORED_DEPTHS" for target in node.targets
        ):
            return [float(value) for value in ast.literal_eval(node.value)]
    raise AssertionError("SCORED_DEPTHS not found in compute_glorys_climatology.py")


def test_scored_depths_include_15m_current_bracket() -> None:
    scored_depths = _scored_depths_from_generation_script()
    assert 13.46714 in scored_depths
    assert 15.81007 in scored_depths
    assert min(scored_depths, key=lambda depth: abs(depth - classIV_support.VELOCITY_TARGET_DEPTH_METERS)) in (
        13.46714,
        15.81007,
    )


@pytest.mark.parametrize("slug", _BASELINE_SLUGS)
def test_baseline_openers_are_registered(slug) -> None:
    assert callable(getattr(challenger_datasets, slug))
    assert callable(getattr(core_challenger_datasets, slug))


@pytest.mark.parametrize("slug", _BASELINE_SLUGS)
def test_runner_enumerates_baselines_like_any_challenger(slug) -> None:
    # run._open_challenger resolves a challenger by attribute on this module; baselines must
    # resolve exactly like the official models (without opening the remote dataset here).
    assert getattr(runner_run.challenger_datasets, slug, None) is not None


def _registry() -> dict:
    return json.loads((_REPOSITORY_ROOT / "challengers.json").read_text(encoding="utf-8"))


def test_challengers_registry_is_schema_valid() -> None:
    validate_against_schema(_registry(), "challengers")


def test_challengers_registry_marks_only_baselines() -> None:
    registry = _registry()
    for slug in _BASELINE_SLUGS:
        assert registry[slug]["is_baseline"] is True
    for slug in _OFFICIAL_SLUGS:
        assert registry[slug]["is_baseline"] is False


def test_challengers_registry_covers_all_openers() -> None:
    registry = _registry()
    assert set(registry) == set(_OFFICIAL_SLUGS) | set(_BASELINE_SLUGS)
    for slug in registry:
        assert callable(getattr(challenger_datasets, slug))


def test_publish_challengers_registry_emits_validated_copy(tmp_path) -> None:
    written_path = publish_challengers_registry(str(tmp_path))
    assert Path(written_path).name == "challengers.json"
    published = json.loads(Path(written_path).read_text(encoding="utf-8"))
    validate_against_schema(published, "challengers")
    assert published == _registry()


@pytest.mark.skipif(
    os.environ.get("OCEANBENCH_LIVE_TESTS") != "1",
    reason="Live remote data check; set OCEANBENCH_LIVE_TESTS=1 to run.",
)
def test_climatology_dataset_opens_remotely() -> None:
    dataset = challenger_datasets.climatology()
    assert Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key() in dataset or "zos" in dataset
    first_field = dataset["zos"] if "zos" in dataset else dataset[Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()]
    sample = first_field.isel({dimension: 0 for dimension in first_field.dims}).compute()
    assert numpy.isfinite(float(sample)) or numpy.isnan(float(sample))
