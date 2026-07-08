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
import pandas
import pytest
import xarray

import oceanbench.core.classIV_support as classIV_support
import oceanbench.core.challenger_datasets as core_challenger_datasets
import oceanbench.datasets.challenger as challenger_datasets
import oceanbench.runner.run as runner_run
from oceanbench.core.classIV import rmsd_class4_validation_per_start
from oceanbench.core.classIV_support import (
    DEFAULT_MEAN_SEA_SURFACE_HEIGHT_SHIFT,
    mean_sea_surface_height_shift,
    prepare_class4_model_variable,
)
from oceanbench.core.dataset_source import get_dataset_source, with_dataset_source
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.schema_validation import validate_against_schema
from oceanbench.publish import publish_challengers_registry
from oceanbench.runner import matchups
from oceanbench.runner.records import RunContext

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
    # The 1-degree variant shares the climatology datum, so its slug resolves to the same shift.
    assert mean_sea_surface_height_shift("climatology_1_degree") == -0.1329
    assert mean_sea_surface_height_shift("persistence") == DEFAULT_MEAN_SEA_SURFACE_HEIGHT_SHIFT
    assert mean_sea_surface_height_shift("glo12") == DEFAULT_MEAN_SEA_SURFACE_HEIGHT_SHIFT
    assert mean_sea_surface_height_shift("glo12_1_degree") == DEFAULT_MEAN_SEA_SURFACE_HEIGHT_SHIFT
    assert mean_sea_surface_height_shift(None) == DEFAULT_MEAN_SEA_SURFACE_HEIGHT_SHIFT
    assert DEFAULT_MEAN_SEA_SURFACE_HEIGHT_SHIFT == -0.1148


_CONSTANT_SEA_SURFACE_HEIGHT = 0.5
_CLIMATOLOGY_SEA_SURFACE_HEIGHT_SHIFT = -0.1329
_FIRST_DAY_DATETIMES = numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]")


def _run_context(challenger_slug: str) -> RunContext:
    return RunContext(
        challenger=challenger_slug,
        challenger_version="0.2.1",
        year=2024,
        region="global",
        oceanbench_version="0.2.1",
    )


def _constant_sea_surface_height_challenger() -> xarray.Dataset:
    """A spatially constant SSH forecast with NO ``oceanbench_source_name`` (i.e. unstaged)."""
    lead_days = numpy.array([0, 1, 2])
    latitudes = numpy.array([0.0, 1.0, 2.0, 3.0])
    longitudes = numpy.array([10.0, 11.0, 12.0, 13.0])
    shape = (len(_FIRST_DAY_DATETIMES), len(lead_days), len(latitudes), len(longitudes))
    return xarray.Dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                numpy.full(shape, _CONSTANT_SEA_SURFACE_HEIGHT),
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): _FIRST_DAY_DATETIMES,
            Dimension.LEAD_DAY_INDEX.key(): lead_days,
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


def _sea_surface_height_observations(observation_value: float) -> xarray.Dataset:
    observation_dimension = "obs"
    times = pandas.to_datetime(["2024-01-03", "2024-01-04", "2024-01-10", "2024-01-12"]).values
    first_days = numpy.array(["2024-01-03", "2024-01-03", "2024-01-10", "2024-01-10"], dtype="datetime64[ns]")
    return xarray.Dataset(
        {
            Dimension.TIME.key(): (observation_dimension, times),
            Dimension.LATITUDE.key(): (observation_dimension, [0.0, 1.0, 2.0, 3.0]),
            Dimension.LONGITUDE.key(): (observation_dimension, [10.0, 11.0, 12.0, 13.0]),
            Dimension.DEPTH.key(): (observation_dimension, [0.0, 0.0, 0.0, 0.0]),
            Dimension.FIRST_DAY_DATETIME.key(): (observation_dimension, first_days),
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (observation_dimension, [observation_value] * 4),
        }
    )


def _patch_mean_dynamic_topography_to_zero(monkeypatch) -> None:
    monkeypatch.setattr(classIV_support, "get_dataset_resolution", lambda dataset: "native")
    monkeypatch.setattr(classIV_support, "load_mean_dynamic_topography", lambda resolution: xarray.DataArray(0.0))


@pytest.mark.parametrize("climatology_slug", ["climatology", "climatology_1_degree"])
def test_climatology_ssh_shift_is_keyed_on_slug_without_staging(monkeypatch, climatology_slug) -> None:
    # Regression guard for the unstaged-climatology fallback: with no local stage the challenger
    # carries no oceanbench_source_name, so the datum shift must resolve from the challenger slug
    # threaded down the runner, not from the (absent) staging-only attribute.
    _patch_mean_dynamic_topography_to_zero(monkeypatch)
    challenger = _constant_sea_surface_height_challenger()
    assert get_dataset_source(challenger) is None  # unstaged

    # With MDT == 0, model SLA == constant_ssh - shift. Anchoring the observation to the
    # climatology-shifted model value makes the climatology slug score exactly 0 and any other
    # challenger score the 0.0181 m gap between the climatology and the default shift.
    observations = _sea_surface_height_observations(
        _CONSTANT_SEA_SURFACE_HEIGHT - _CLIMATOLOGY_SEA_SURFACE_HEIGHT_SHIFT
    )
    variables = [Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID]

    matchup_frame = matchups.class4_matchups(
        challenger, observations, variables, context=_run_context(climatology_slug)
    )
    numpy.testing.assert_allclose(matchup_frame["sla_shift"].to_numpy(), _CLIMATOLOGY_SEA_SURFACE_HEIGHT_SHIFT)
    numpy.testing.assert_allclose(
        matchup_frame["model_value"].to_numpy(),
        _CONSTANT_SEA_SURFACE_HEIGHT - _CLIMATOLOGY_SEA_SURFACE_HEIGHT_SHIFT,
    )

    score = rmsd_class4_validation_per_start(challenger, observations, variables, challenger_slug=climatology_slug)
    numpy.testing.assert_allclose(score["rmsd"].to_numpy(), 0.0, atol=1e-9)


def test_default_ssh_shift_untouched_for_non_climatology_slug(monkeypatch) -> None:
    _patch_mean_dynamic_topography_to_zero(monkeypatch)
    challenger = _constant_sea_surface_height_challenger()
    observations = _sea_surface_height_observations(
        _CONSTANT_SEA_SURFACE_HEIGHT - _CLIMATOLOGY_SEA_SURFACE_HEIGHT_SHIFT
    )
    variables = [Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID]

    matchup_frame = matchups.class4_matchups(challenger, observations, variables, context=_run_context("glonet"))
    numpy.testing.assert_allclose(matchup_frame["sla_shift"].to_numpy(), DEFAULT_MEAN_SEA_SURFACE_HEIGHT_SHIFT)

    score = rmsd_class4_validation_per_start(challenger, observations, variables, challenger_slug="glo12")
    expected_gap = abs(DEFAULT_MEAN_SEA_SURFACE_HEIGHT_SHIFT - _CLIMATOLOGY_SEA_SURFACE_HEIGHT_SHIFT)
    numpy.testing.assert_allclose(score["rmsd"].to_numpy(), expected_gap, atol=1e-9)


def test_staged_climatology_ssh_shift_unchanged(monkeypatch) -> None:
    # No regression when staged: the source attribute alone (no slug) still yields the climatology shift.
    _patch_mean_dynamic_topography_to_zero(monkeypatch)
    staged_challenger = with_dataset_source(
        _constant_sea_surface_height_challenger(), kind="challenger", name="climatology"
    )
    observations = _sea_surface_height_observations(
        _CONSTANT_SEA_SURFACE_HEIGHT - _CLIMATOLOGY_SEA_SURFACE_HEIGHT_SHIFT
    )
    score = rmsd_class4_validation_per_start(staged_challenger, observations, [Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID])
    numpy.testing.assert_allclose(score["rmsd"].to_numpy(), 0.0, atol=1e-9)


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
