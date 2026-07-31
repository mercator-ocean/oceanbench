# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Local evaluation end to end (contracts.md §7): overlay agreement, self-containedness, scorecard.

The key proof: scoring a forecast against an offline reference bundle reproduces the published
per-start values of the same model exactly (the overlay shows the user's model and the published
challenger in agreement). Exercised here on synthetic, network-free data; the real
glonet_1_degree run is in ``test_evaluate_local_real.py`` (skipped by default).
"""

import json
from pathlib import Path

import pandas
import pytest
import xarray

from oceanbench.cli import _build_parser
from oceanbench.packs.evaluate import (
    load_pack_manifest,
    open_forecast_dataset,
    per_start_agreement,
    _pack_observation_opener,
    _pack_reference_opener,
    evaluate,
)

_AGREEMENT_TOLERANCE = 1e-9


@pytest.fixture
def restored_runtime_configuration():
    """Keep a test that installs a global runtime configuration from leaking it into the next one."""
    from oceanbench.core import runtime_configuration as runtime_configuration_module

    previous = runtime_configuration_module._runtime_configuration
    yield
    runtime_configuration_module._runtime_configuration = previous


def test_evaluate_cli_has_only_the_approved_surface():
    parser = _build_parser()
    arguments = parser.parse_args(["evaluate", "forecast.zarr", "--metrics", "rmsd", "lagrangian"])
    assert arguments.metrics == ["rmsd", "lagrangian"]
    evaluate_parser = next(
        action.choices["evaluate"]
        for action in parser._actions
        if getattr(action, "choices", None) and "evaluate" in action.choices
    )
    option_strings = {option for action in evaluate_parser._actions for option in action.option_strings}
    assert option_strings == {
        "-h",
        "--help",
        "--output",
        "--region",
        "--year",
        "--metrics",
        "--viewer-artifacts",
        "--offline-references",
        "--s3-bucket",
        "--s3-prefix",
        "--s3-endpoint",
        "--s3-env-file",
        # Restored from the 0.4.0 CLI so an old command line still runs, on the new route.
        "--all-challengers",
        "--region-file",
        "--cache-dir",
        "--stage",
        "--stage-dir",
        "--stage-max-workers",
        "--remote-retries",
        "--keep-stage",
        "--output-bucket",
        "--output-prefix",
        "--max-workers",
    }


def test_scores_are_the_only_default_output():
    """Nothing viewer-related is built unless it is asked for (contracts.md §7)."""
    arguments = _build_parser().parse_args(["evaluate", "forecast.zarr"])
    assert arguments.viewer_artifacts is False
    assert arguments.offline_references is None
    assert arguments.region is None
    assert arguments.year is None


def test_a_registered_challenger_slug_is_accepted_as_the_target():
    arguments = _build_parser().parse_args(["evaluate", "glonet_1_degree"])
    assert arguments.target == ["glonet_1_degree"]


def test_a_zero_four_command_line_still_parses():
    arguments = _build_parser().parse_args(
        [
            "evaluate",
            "challenger_datasets/glonet.py",
            "--output-bucket",
            "project-oceanbench",
            "--output-prefix",
            "dev/reports",
            "--max-workers",
            "4",
            "--stage",
            "all",
            "--stage-dir",
            "/scratch/stage",
            "--stage-max-workers",
            "8",
            "--region",
            "ibi",
        ]
    )
    assert arguments.target == ["challenger_datasets/glonet.py"]
    assert arguments.output_bucket == "project-oceanbench"
    assert arguments.output_prefix == "dev/reports"
    assert arguments.stage == ["all"]


def test_stage_and_cache_flags_reach_the_runtime_configuration(restored_runtime_configuration):
    from oceanbench.cli import _apply_runtime_configuration
    from oceanbench.core.runtime_configuration import current_runtime_configuration

    arguments = _build_parser().parse_args(
        ["evaluate", "forecast.zarr", "--stage", "observations", "--stage-dir", "/scratch/stage", "--cache-dir", "/c"]
    )
    _apply_runtime_configuration(arguments)
    configuration = current_runtime_configuration()

    assert configuration.staged_components == ("observations",)
    assert configuration.stage_directory == "/scratch/stage"
    assert str(configuration.local_cache_directory()) == "/c"


def test_the_chunk_cache_stays_off_when_no_cache_directory_is_given(monkeypatch, restored_runtime_configuration):
    from oceanbench.cli import _apply_runtime_configuration
    from oceanbench.core.runtime_configuration import current_runtime_configuration

    monkeypatch.delenv("OCEANBENCH_LOCAL_CACHE", raising=False)
    arguments = _build_parser().parse_args(["evaluate", "forecast.zarr"])
    _apply_runtime_configuration(arguments)

    assert current_runtime_configuration().local_cache_directory() is None


def test_a_python_challenger_file_is_opened_by_its_challenger_dataset_variable(tmp_path):
    from oceanbench.packs.evaluate import open_python_challenger_file

    challenger_path = tmp_path / "challenger.py"
    challenger_path.write_text(
        'import xarray\n\nchallenger_dataset = xarray.Dataset({"thetao": ("x", [1.0, 2.0])})\n',
        encoding="utf-8",
    )

    dataset = open_python_challenger_file(str(challenger_path))

    assert list(dataset.data_vars) == ["thetao"]


def test_a_python_challenger_file_without_the_variable_is_rejected(tmp_path):
    from oceanbench.packs.evaluate import open_python_challenger_file

    challenger_path = tmp_path / "challenger.py"
    challenger_path.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="challenger_dataset"):
        open_python_challenger_file(str(challenger_path))


def test_a_python_challenger_target_prints_the_new_pipeline_notice(monkeypatch, capsys, restored_runtime_configuration):
    from oceanbench.cli import NEW_PIPELINE_NOTICE, _run_evaluate

    arguments = _build_parser().parse_args(["evaluate", "/missing/challenger.py"])
    monkeypatch.setattr("oceanbench.cli._evaluate_one_target", lambda *args, **kwargs: 0)

    assert _run_evaluate(arguments) == 0
    assert capsys.readouterr().err.splitlines() == [NEW_PIPELINE_NOTICE]


def test_all_challengers_expands_to_the_registered_slugs(monkeypatch, capsys, restored_runtime_configuration):
    from oceanbench.cli import NEW_PIPELINE_NOTICE, _run_evaluate
    from oceanbench.runner.run import registered_challengers

    scored = []
    arguments = _build_parser().parse_args(["evaluate", "--all-challengers"])
    monkeypatch.setattr(
        "oceanbench.cli._evaluate_one_target", lambda _arguments, target, **kwargs: scored.append(target) or 0
    )

    assert _run_evaluate(arguments) == 0
    assert scored == list(registered_challengers())
    assert capsys.readouterr().err.splitlines() == [NEW_PIPELINE_NOTICE]


def test_an_unknown_target_lists_the_accepted_forms_and_the_nearest_slug():
    from oceanbench.packs.evaluate import _open_evaluation_target

    with pytest.raises(ValueError) as error:
        _open_evaluation_target("glonett")

    message = str(error.value)
    assert "a registered challenger slug" in message
    assert "forecast zarr" in message
    assert ".py file" in message
    assert "Did you mean 'glonet'?" in message


def _run(fixture, tmp_path, **overrides):
    options = dict(
        offline_references_directory=fixture.offline_references_directory,
        output_directory=str(tmp_path / "out"),
        published_scores_path=fixture.published_scores_path,
        published_challengers_path=fixture.published_challengers_path,
        metrics=("rmsd",),
    )
    options.update(overrides)
    return evaluate(fixture.forecast_path, **options)


def test_live_edito_sources_are_the_default(local_evaluation_fixture, tmp_path, monkeypatch):
    """No offline bundle means the live openers, the default year and the default region."""
    consulted_references = []

    def _live_glorys(_challenger_dataset):
        consulted_references.append("glorys")
        return local_evaluation_fixture.full_depth_reference_dataset

    def _refuse_to_open_the_observations(_challenger_dataset):
        raise AssertionError("a scores-only run must not reach for the observations")

    monkeypatch.setattr("oceanbench.packs.evaluate.LIVE_REFERENCE_OPENERS", {"glorys": _live_glorys})
    monkeypatch.setattr("oceanbench.packs.evaluate.live_observation_opener", _refuse_to_open_the_observations)

    result = evaluate(
        local_evaluation_fixture.forecast_path,
        output_directory=str(tmp_path / "out"),
        published_scores_path=local_evaluation_fixture.published_scores_path,
        published_challengers_path=local_evaluation_fixture.published_challengers_path,
        metrics=("rmsd",),
    )

    assert set(consulted_references) == {"glorys"}
    scores = pandas.read_parquet(result.scores_path)
    assert set(scores["year"]) == {2024}
    assert set(scores["region"]) == {"global"}
    # Live sources carry every depth, unlike a quick bundle's surface-only references.
    assert set(scores.loc[scores["metric"] == "rmsd", "depth"].dropna()) > {"surface"}


def test_region_or_year_contradicting_the_offline_bundle_is_rejected(local_evaluation_fixture, tmp_path):
    """The bundle's manifest fixes the evaluation context; a contradiction is an error, not a shrug."""
    with pytest.raises(ValueError, match="region"):
        _run(local_evaluation_fixture, tmp_path, region="ibi")
    with pytest.raises(ValueError, match="year"):
        _run(local_evaluation_fixture, tmp_path, year=2023)


def test_overlay_agrees_with_published_per_start(local_evaluation_fixture, tmp_path):
    result = _run(local_evaluation_fixture, tmp_path)
    published = pandas.read_parquet(local_evaluation_fixture.published_scores_path)

    agreement = per_start_agreement(result.scores, published)
    assert not agreement.empty
    assert agreement["absolute_difference"].max() < _AGREEMENT_TOLERANCE


def test_pack_is_self_contained_every_reference_resolves_from_the_manifest(local_evaluation_fixture):
    pack_directory = Path(local_evaluation_fixture.offline_references_directory)
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
    from oceanbench.runner.records import DIAGNOSTIC_METRICS

    result = _run(local_evaluation_fixture, tmp_path, metrics=("rmsd",))
    scores = pandas.read_parquet(result.scores_path)
    assert set(scores["metric"]).difference(DIAGNOSTIC_METRICS) == {"rmsd"}


def test_grid_coverage_travels_with_the_scores_but_never_reaches_the_summary(local_evaluation_fixture, tmp_path):
    """A run that scored a snapped grid says so in its own scores file (issue #305)."""
    result = _run(local_evaluation_fixture, tmp_path, metrics=("rmsd",))
    scores = pandas.read_parquet(result.scores_path)

    coverage = scores[scores["metric"] == "grid_coverage"]
    assert set(coverage["reference"]) == {"glorys"}
    assert (coverage["value"] > 0.999).all()
    assert (coverage["n"] > 0).all()
    # One row per reference, not one per metric family sharing that reference.
    assert len(coverage) == 1

    summary = json.loads(Path(result.summary_path).read_text())
    assert "grid_coverage" not in {record["metric"] for record in summary}


def test_year_and_region_are_inferred_from_manifest(local_evaluation_fixture, tmp_path):
    result = _run(local_evaluation_fixture, tmp_path)
    scores = pandas.read_parquet(result.scores_path)
    assert set(scores["year"]) == {2024}
    assert set(scores["region"]) == {"global"}


def test_open_forecast_accepts_a_combined_zarr(local_evaluation_fixture):
    dataset = open_forecast_dataset(local_evaluation_fixture.forecast_path)
    assert "first_day_datetime" in dataset.dims
    assert "sea_surface_height_above_geoid" in dataset.data_vars


def test_viewer_builds_local_pyramid_and_mixed_catalog(local_evaluation_fixture, tmp_path, monkeypatch):
    remote = {
        "slug": "official",
        "label": "Official",
        "store": "https://example.test/official.zarr",
        "manifest": "https://example.test/official.viewer-manifest.json",
    }
    monkeypatch.setattr("oceanbench.packs.local_viewer._official_datasets", lambda: [remote])

    result = _run(local_evaluation_fixture, tmp_path, viewer_artifacts=True)

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


def test_viewer_keeps_scores_and_adds_the_map(local_evaluation_fixture, tmp_path, monkeypatch):
    monkeypatch.setattr("oceanbench.packs.local_viewer._official_datasets", lambda: [])

    result = _run(local_evaluation_fixture, tmp_path, viewer_artifacts=True)

    assert Path(result.scores_path).exists()
    assert Path(result.viewer_zarr_path, ".zmetadata").exists()


def test_viewer_site_adopts_the_pyramid_the_serving_artifacts_already_built(
    local_evaluation_fixture, tmp_path, monkeypatch
):
    """One pyramid, not two: both writers target viewer/data/<slug>.zarr (contracts.md §5)."""
    from oceanbench.publish.viewer_artifacts import ViewerArtifactsResult

    monkeypatch.setattr("oceanbench.packs.local_viewer._official_datasets", lambda: [])
    served = ViewerArtifactsResult(
        pyramid_zarr_path="/already/built/your_model.zarr",
        pyramid_manifest_path="/already/built/your_model.viewer-manifest.json",
    )
    monkeypatch.setattr("oceanbench.publish.viewer_artifacts.write_viewer_artifacts", lambda **_: served)

    def _refuse_to_build_a_second_pyramid(*_args, **_keywords):
        raise AssertionError("the pyramid was rebuilt instead of adopted")

    monkeypatch.setattr("oceanbench.packs.local_viewer.build_pyramid", _refuse_to_build_a_second_pyramid)

    result = _run(local_evaluation_fixture, tmp_path, viewer_artifacts=True)

    assert result.viewer_zarr_path == served.pyramid_zarr_path
    assert result.viewer_manifest_path == served.pyramid_manifest_path


def test_the_map_still_builds_when_the_serving_artifacts_fail(local_evaluation_fixture, tmp_path, monkeypatch):
    """A missing observation store must cost the insights panels, not the whole viewer."""
    monkeypatch.setattr("oceanbench.packs.local_viewer._official_datasets", lambda: [])

    def _fail(**_keywords):
        raise RuntimeError("no observations here")

    monkeypatch.setattr("oceanbench.publish.viewer_artifacts.write_viewer_artifacts", _fail)

    result = _run(local_evaluation_fixture, tmp_path, viewer_artifacts=True)

    assert any("viewer serving artifacts skipped" in flag for flag in result.flags)
    assert Path(result.viewer_zarr_path, ".zmetadata").exists()


def test_scoring_a_published_challenger_needs_no_forecast_path(local_evaluation_fixture, tmp_path, monkeypatch):
    """A registered slug is opened by the library itself, and gets no overlay scorecard."""
    import xarray

    from oceanbench.packs.evaluate import evaluate

    forecast = xarray.open_dataset(local_evaluation_fixture.forecast_path, engine="zarr")
    monkeypatch.setattr("oceanbench.packs.evaluate.open_registered_challenger", lambda _slug: forecast)

    result = evaluate(
        "glonet_1_degree",
        offline_references_directory=local_evaluation_fixture.offline_references_directory,
        output_directory=str(tmp_path / "out"),
        metrics=("rmsd",),
    )

    scores = pandas.read_parquet(result.scores_path)
    assert set(scores["challenger"]) == {"glonet_1_degree"}
    assert result.scorecard_path is None
