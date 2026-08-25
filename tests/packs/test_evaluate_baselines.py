# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Skill-vs-baseline offline: the pack's bundled baselines are scored and paired with the model.

Synthetic and network-free, on the same fake pack the rest of the local-evaluation tests use.
A pack built before baselines existed carries none, so the no-baselines path is the graceful
degradation every older quick pack takes.
"""

import json
from pathlib import Path
import shutil

import pandas
import pytest

from oceanbench.packs.evaluate import (
    NO_BUNDLED_BASELINES_FLAG,
    evaluate,
    skill_baseline_slug,
)

from tests.packs.conftest import _synthetic_dataset, _write_zarr


def _bundle_baseline(pack_directory: Path, slug: str, *, seed: int, starts: slice | None = None) -> None:
    """Add a baseline store to a fake offline reference directory and register it in the manifest."""
    baseline = _synthetic_dataset(seed=seed, offset=0.6)
    if starts is not None:
        baseline = baseline.isel(first_day_datetime=starts)
    _write_zarr(baseline, pack_directory / "baselines" / f"{slug}.zarr")

    manifest_path = pack_directory / "pack-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contents"]["baselines"][slug] = {
        "path": f"baselines/{slug}.zarr",
        "variables": sorted(str(name) for name in baseline.data_vars),
        "depths": ["surface"],
    }
    manifest["baselines_available"] = True
    manifest["notes"] = []
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")


@pytest.fixture
def pack_with_climatology(local_evaluation_fixture):
    pack_directory = Path(local_evaluation_fixture.offline_references_directory)
    _bundle_baseline(pack_directory, "climatology", seed=7)
    return local_evaluation_fixture


def _evaluate(fixture, output_directory: Path):
    return evaluate(
        fixture.forecast_path,
        output_directory=str(output_directory),
        offline_references_directory=fixture.offline_references_directory,
        published_scores_path=fixture.published_scores_path,
        published_challengers_path=fixture.published_challengers_path,
        metrics=("rmsd",),
    )


def test_skill_baseline_slug_prefers_climatology():
    assert skill_baseline_slug({"persistence", "climatology"}) == "climatology"
    assert skill_baseline_slug({"persistence"}) == "persistence"
    assert skill_baseline_slug({"climatology_1_degree", "persistence_1_degree"}) == "climatology_1_degree"
    assert skill_baseline_slug({"some_other_baseline"}) == "some_other_baseline"
    assert skill_baseline_slug(set()) is None


def test_bundled_baseline_is_scored_from_the_pack(pack_with_climatology, tmp_path):
    result = _evaluate(pack_with_climatology, tmp_path / "out")

    assert result.skill_baseline == "climatology"
    assert not result.baseline_scores.empty
    assert set(result.baseline_scores["challenger"].unique()) == {"climatology"}
    assert NO_BUNDLED_BASELINES_FLAG not in result.flags


def test_the_returned_scores_stay_the_targets_own(pack_with_climatology, tmp_path):
    result = _evaluate(pack_with_climatology, tmp_path / "out")

    assert set(result.scores["challenger"].unique()) == {"your_model"}


def test_scores_parquet_carries_the_model_and_the_baseline(pack_with_climatology, tmp_path):
    result = _evaluate(pack_with_climatology, tmp_path / "out")

    published = pandas.read_parquet(result.scores_path)
    assert set(published["challenger"].unique()) == {"your_model", "climatology"}


def test_summary_carries_skill_against_the_bundled_baseline(pack_with_climatology, tmp_path):
    result = _evaluate(pack_with_climatology, tmp_path / "out")

    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    assert summary
    skill_column = "skill_vs_climatology"
    assert all(skill_column in row for row in summary)
    your_model_rows = [row for row in summary if row["challenger"] == "your_model"]
    assert your_model_rows
    assert any(row[skill_column] is not None for row in your_model_rows)
    # The baseline against itself is exactly zero skill; that identity proves the pairing is real.
    baseline_rows = [row for row in summary if row["challenger"] == "climatology"]
    assert baseline_rows
    assert all(row[skill_column] == pytest.approx(0.0) for row in baseline_rows)


def test_two_bundled_baselines_are_both_scored(local_evaluation_fixture, tmp_path):
    pack_directory = Path(local_evaluation_fixture.offline_references_directory)
    _bundle_baseline(pack_directory, "climatology", seed=7)
    _bundle_baseline(pack_directory, "persistence", seed=8)

    result = _evaluate(local_evaluation_fixture, tmp_path / "out")

    assert set(result.baseline_scores["challenger"].unique()) == {"climatology", "persistence"}
    assert result.skill_baseline == "climatology"


def test_a_pack_without_baselines_scores_without_skill_and_says_so(local_evaluation_fixture, tmp_path):
    result = _evaluate(local_evaluation_fixture, tmp_path / "out")

    assert result.skill_baseline is None
    assert result.baseline_scores.empty
    assert NO_BUNDLED_BASELINES_FLAG in result.flags
    assert not result.scores.empty

    published = pandas.read_parquet(result.scores_path)
    assert set(published["challenger"].unique()) == {"your_model"}
    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    assert summary
    assert not any(key.startswith("skill_vs_") for row in summary for key in row)


def test_a_baseline_sharing_no_start_is_flagged_and_dropped(local_evaluation_fixture, tmp_path):
    pack_directory = Path(local_evaluation_fixture.offline_references_directory)
    _bundle_baseline(pack_directory, "climatology", seed=7, starts=slice(0, 0))

    result = _evaluate(local_evaluation_fixture, tmp_path / "out")

    assert result.skill_baseline is None
    assert any("shares no start date" in flag for flag in result.flags)
    assert not result.scores.empty


def test_an_unreadable_baseline_never_aborts_the_run(local_evaluation_fixture, tmp_path):
    pack_directory = Path(local_evaluation_fixture.offline_references_directory)
    _bundle_baseline(pack_directory, "climatology", seed=7)
    # The manifest still promises the store; the store itself is gone, as a truncated fetch leaves it.
    shutil.rmtree(pack_directory / "baselines" / "climatology.zarr")

    result = _evaluate(local_evaluation_fixture, tmp_path / "out")

    assert result.skill_baseline is None
    assert any(flag.startswith("baseline climatology skipped:") for flag in result.flags)
    assert not result.scores.empty


def test_the_live_path_never_looks_for_bundled_baselines(local_evaluation_fixture, tmp_path, monkeypatch):
    from oceanbench.runner import run as run_module

    reference = local_evaluation_fixture.full_depth_reference_dataset
    monkeypatch.setattr(run_module, "LIVE_REFERENCE_OPENERS", {"glorys": lambda challenger: reference})
    monkeypatch.setattr("oceanbench.packs.evaluate.LIVE_REFERENCE_OPENERS", {"glorys": lambda challenger: reference})

    result = evaluate(
        local_evaluation_fixture.forecast_path,
        output_directory=str(tmp_path / "out"),
        metrics=("rmsd",),
    )

    assert result.skill_baseline is None
    assert result.baseline_scores.empty
    assert NO_BUNDLED_BASELINES_FLAG not in result.flags
