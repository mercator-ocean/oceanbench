# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Local evaluation end to end (contracts.md §7): overlay agreement, self-containedness, scorecard.

The key proof: scoring a forecast against a pack reproduces the published per-start values of the
same model exactly (the overlay shows the user's model and the published challenger in agreement).
Exercised here on synthetic, network-free data; the real glonet_1_degree run is in
``test_evaluate_local_real.py`` (skipped by default).
"""

import json
from pathlib import Path

import pandas
import xarray

from oceanbench.cli import _build_parser, main
from oceanbench.packs.evaluate import (
    load_pack_manifest,
    open_forecast_dataset,
    per_start_agreement,
    _pack_observation_opener,
    _pack_reference_opener,
    evaluate_local,
)

_AGREEMENT_TOLERANCE = 1e-9


def test_evaluate_local_cli_has_only_the_approved_surface():
    parser = _build_parser()
    arguments = parser.parse_args(["evaluate", "forecast.zarr", "--pack", "pack", "--metrics", "rmsd", "lagrangian"])
    assert arguments.metrics == ["rmsd", "lagrangian"]
    evaluate_local_parser = next(
        action.choices["evaluate"]
        for action in parser._actions
        if getattr(action, "choices", None) and "evaluate" in action.choices
    )
    option_strings = {option for action in evaluate_local_parser._actions for option in action.option_strings}
    assert option_strings == {"-h", "--help", "--pack", "--output", "--artifacts", "--metrics"}


def test_evaluate_local_is_a_hidden_deprecated_alias(monkeypatch, capsys):
    parser = _build_parser()
    assert "evaluate-local" not in parser.format_help()
    monkeypatch.setattr(
        "sys.argv",
        ["oceanbench", "evaluate-local", "forecast.zarr", "--pack", "pack"],
    )
    monkeypatch.setattr("oceanbench.cli._run_evaluate", lambda _args: 0)

    try:
        main()
    except SystemExit as exit:
        assert exit.code == 0
    else:
        raise AssertionError("Expected CLI entry point to exit")

    assert capsys.readouterr().err == ("note: 'oceanbench evaluate-local' is deprecated; use 'oceanbench evaluate'\n")


def _run(fixture, tmp_path, **overrides):
    options = dict(
        pack_directory=fixture.pack_directory,
        output_directory=str(tmp_path / "out"),
        published_scores_path=fixture.published_scores_path,
        published_challengers_path=fixture.published_challengers_path,
        metrics=("rmsd",),
    )
    options.update(overrides)
    return evaluate_local(fixture.forecast_path, **options)


def test_overlay_agrees_with_published_per_start(local_evaluation_fixture, tmp_path):
    result = _run(local_evaluation_fixture, tmp_path)
    published = pandas.read_parquet(local_evaluation_fixture.published_scores_path)

    agreement = per_start_agreement(result.scores, published)
    assert not agreement.empty
    assert agreement["absolute_difference"].max() < _AGREEMENT_TOLERANCE


def test_pack_is_self_contained_every_reference_resolves_from_the_manifest(local_evaluation_fixture):
    pack_directory = Path(local_evaluation_fixture.pack_directory)
    manifest = load_pack_manifest(str(pack_directory))

    for reference_entry in manifest["contents"]["references"].values():
        assert (pack_directory / reference_entry["path"]).exists()
        opener = _pack_reference_opener(pack_directory, reference_entry["path"])
        resolved = opener(xarray.Dataset())
        assert isinstance(resolved, xarray.Dataset)
        assert set(reference_entry["variables"]).issubset(set(resolved.data_vars))

    observation_path = manifest["contents"]["observations"]["path"]
    assert (pack_directory / observation_path).exists()
    observation_opener = _pack_observation_opener(pack_directory, observation_path)
    assert observation_opener is not None


def test_emits_records_parquet_and_aggregated_summary(local_evaluation_fixture, tmp_path):
    result = _run(local_evaluation_fixture, tmp_path)

    scores = pandas.read_parquet(result.scores_path)
    assert "rmsd" in set(scores["metric"])
    # Quick pack scores surface only: no subsurface depth labels leak through.
    gridded_depths = set(scores.loc[scores["metric"] == "rmsd", "depth"].dropna())
    assert gridded_depths <= {"surface"}

    summary = json.loads(Path(result.summary_path).read_text())
    assert summary
    assert {record["challenger"] for record in summary} == {"your_model"}


def test_scorecard_is_self_contained_and_overlays_your_model(local_evaluation_fixture, tmp_path):
    result = _run(local_evaluation_fixture, tmp_path)
    html = Path(result.scorecard_path).read_text()

    # Data is inlined (no fetch / no ES module), so the report opens over file:// with no server.
    assert 'id="scorecard-data"' in html
    assert 'getElementById("scorecard-data")' in html
    assert 'fetch("' not in html and "fetch('" not in html
    assert 'type="module"' not in html
    # Both the user's model and the published challenger are present in the overlay payload.
    assert '"your_model"' in html
    assert f'"{local_evaluation_fixture.published_challenger_slug}"' in html


def test_metrics_selects_only_requested_metric_family(local_evaluation_fixture, tmp_path):
    result = _run(local_evaluation_fixture, tmp_path, metrics=("rmsd",))
    scores = pandas.read_parquet(result.scores_path)
    assert set(scores["metric"]) == {"rmsd"}


def test_year_and_region_are_inferred_from_manifest(local_evaluation_fixture, tmp_path):
    result = _run(local_evaluation_fixture, tmp_path)
    scores = pandas.read_parquet(result.scores_path)
    assert set(scores["year"]) == {2024}
    assert set(scores["region"]) == {"global"}


def test_open_forecast_accepts_a_combined_zarr(local_evaluation_fixture):
    dataset = open_forecast_dataset(local_evaluation_fixture.forecast_path)
    assert "first_day_datetime" in dataset.dims
    assert "sea_surface_height_above_geoid" in dataset.data_vars


def test_all_artifacts_builds_local_pyramid_and_mixed_catalog(local_evaluation_fixture, tmp_path, monkeypatch):
    remote = {
        "slug": "official",
        "label": "Official",
        "store": "https://example.test/official.zarr",
        "manifest": "https://example.test/official.viewer-manifest.json",
    }
    monkeypatch.setattr("oceanbench.packs.local_viewer._official_datasets", lambda: [remote])

    result = _run(local_evaluation_fixture, tmp_path, artifacts="all")

    assert Path(result.scores_path).exists()
    assert Path(result.viewer_zarr_path, ".zmetadata").exists()
    manifest = json.loads(Path(result.viewer_manifest_path).read_text())
    assert manifest["dataset"] == "your_model"
    catalog = json.loads(Path(result.viewer_datasets_path).read_text())
    assert catalog["datasets"][0]["store"] == "./data/your_model.zarr"
    assert catalog["datasets"][1] == remote
    assert Path(result.viewer_directory, "index.html").exists()
    assert Path(result.viewer_directory, "data", "insights.json").exists()
    assert 'rel="icon" href="data:image/svg+xml' in Path(result.viewer_directory, "index.html").read_text()


def test_all_artifacts_keeps_scores_and_adds_viewer(local_evaluation_fixture, tmp_path, monkeypatch):
    monkeypatch.setattr("oceanbench.packs.local_viewer._official_datasets", lambda: [])

    result = _run(local_evaluation_fixture, tmp_path, artifacts="all")

    assert Path(result.scores_path).exists()
    assert Path(result.viewer_zarr_path, ".zmetadata").exists()
