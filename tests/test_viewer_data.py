# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json

import numpy
import pandas
import pytest
import xarray

import oceanbench
import oceanbench.core.evaluate as evaluate
from oceanbench.core.classIV import class4_rmsd_table
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.regions import GLOBAL, IBI
from oceanbench.core.scores import CLASS4_METRIC_FUNCTION_NAME, METRIC_FUNCTION_NAMES, score_rows
from oceanbench.publish import viewer_data
from oceanbench.publish.viewer_artifacts import ViewerArtifacts, class4_rmsd_rows_by_start
from tests import synthetic_class4
from tests.test_classiv_matchups import _matchups
from tests.test_evaluate import evaluated_metrics  # noqa: F401
from tests.test_viewer_artifacts import _frame, _sea_surface_height_dataset

_START_DATES = [pandas.Timestamp(first_day) for first_day in synthetic_class4.FIRST_DAYS]


def _class4_summary_rows() -> tuple[list[dict], list[dict]]:
    matchups = _matchups()
    document_rows = score_rows(
        CLASS4_METRIC_FUNCTION_NAME, class4_rmsd_table(matchups, synthetic_class4.LEAD_DAYS_COUNT)
    )
    class4_by_start = viewer_data.class4_scores_by_start(class4_rmsd_rows_by_start(matchups), _START_DATES)
    return document_rows, viewer_data.scores_summary_rows(
        "your_model", GLOBAL, document_rows, _START_DATES, {}, class4_by_start
    )


def test_class4_summary_means_are_the_official_scores_and_the_starts_recombine_to_them() -> None:
    matchups = _matchups()
    class4_by_start = viewer_data.class4_scores_by_start(class4_rmsd_rows_by_start(matchups), _START_DATES)
    document_rows, summary_rows = _class4_summary_rows()

    assert len(summary_rows) == len(document_rows)
    for document_row, summary_row in zip(document_rows, summary_rows):
        assert summary_row["mean"] == document_row["value"]
        assert (summary_row["metric"], summary_row["reference"]) == ("class4_rmsd", "observations")
        if document_row["value"] is None:
            assert summary_row["ci_low"] is None and summary_row["n_starts"] == 0
            continue
        values, counts = class4_by_start[
            (
                "rmsd_of_variables",
                "observations",
                document_row["standard_name"],
                document_row["depth"],
                document_row["lead_day"],
            )
        ]
        present = counts > 0
        # The official Class IV score pools the match-ups, so the per-start scores recombine to it.
        recombined = numpy.sqrt((values[present] ** 2 * counts[present]).sum() / counts[present].sum())
        numpy.testing.assert_allclose(recombined, document_row["value"], rtol=1e-12)
        assert summary_row["n"] == counts.sum()
        assert summary_row["n_starts"] == present.sum()
        assert summary_row["ci_low"] <= summary_row["ci_high"]
    assert _class4_summary_rows()[1] == summary_rows


def test_gridded_scores_by_start_run_the_public_metrics_on_one_start_at_a_time(monkeypatch) -> None:
    calls = []

    def metric(challenger_dataset, region):
        calls.append(len(challenger_dataset["first_day_datetime"]))
        value = float(challenger_dataset["first_day_datetime"].dt.day.values[0])
        return pandas.DataFrame(
            [[value, 2 * value]],
            index=["Salinity (PSU) [sea_water_salinity]{surface}"],
            columns=lead_day_labels(1, 2),
        )

    for metric_function_name in viewer_data.GRIDDED_METRIC_FUNCTION_NAMES:
        monkeypatch.setattr(oceanbench.metrics, metric_function_name, metric)

    scores = viewer_data.gridded_scores_by_start(synthetic_class4.challenger_dataset(), GLOBAL)

    assert calls == [1] * 2 * len(viewer_data.GRIDDED_METRIC_FUNCTION_NAMES)
    assert CLASS4_METRIC_FUNCTION_NAME not in viewer_data.GRIDDED_METRIC_FUNCTION_NAMES
    numpy.testing.assert_array_equal(
        scores[("rmsd_of_variables", "glorys_reanalysis", "sea_water_salinity", "surface", 2)], [6.0, 20.0]
    )


def test_gridded_summary_rows_report_the_official_score_with_a_start_bootstrap_interval() -> None:
    document_rows = [
        {
            "metric": "rmsd_of_mixed_layer_depth",
            "reference": "glo12_analysis",
            "variable": "mixed layer depth",
            "standard_name": "ocean_mixed_layer_thickness",
            "depth": "surface",
            "unit": "m",
            "lead_day": 1,
            "value": 2.5,
        },
        {
            "metric": "deviation_of_lagrangian_trajectories",
            "reference": "glorys_reanalysis",
            "variable": "surface lagrangian trajectory deviation",
            "standard_name": None,
            "depth": "surface",
            "unit": "km",
            "lead_day": 1,
            "value": 10.0,
        },
    ]
    gridded_by_start = {
        ("rmsd_of_mixed_layer_depth", "glo12_analysis", "ocean_mixed_layer_thickness", "surface", 1): numpy.array(
            [2.0, 3.0]
        )
    }

    (row,) = viewer_data.scores_summary_rows("your_model", IBI, document_rows, _START_DATES, gridded_by_start, {})

    assert (row["metric"], row["reference"], row["variable"], row["depth"]) == (
        "rmsd",
        "glo12",
        "ocean_mixed_layer_thickness",
        None,
    )
    assert (row["region"], row["year"], row["mean"], row["n_starts"], row["n"]) == ("ibi", 2024, 2.5, 2, None)
    assert 2.0 <= row["ci_low"] <= row["ci_high"] <= 3.0


def _gridded_metric(challenger_dataset, region):
    return pandas.DataFrame(
        [[1.0, 2.0]], index=["Salinity (PSU) [sea_water_salinity]{surface}"], columns=lead_day_labels(1, 2)
    )


def test_global_viewer_data_holds_the_maps_the_insights_and_the_indexes(tmp_path, monkeypatch) -> None:
    for metric_function_name in viewer_data.GRIDDED_METRIC_FUNCTION_NAMES:
        monkeypatch.setattr(oceanbench.metrics, metric_function_name, _gridded_metric)
    start_dates = ["2024-01-03", "2024-01-10"]
    dataset = _sea_surface_height_dataset(tuple(start_dates))
    profile = xarray.DataArray([1.0, 0.9, 0.5], coords={"depth": [0.5, 10.0, 50.0]})
    surface = xarray.ones_like(dataset["sea_surface_height_above_geoid"])
    dataset["sea_water_potential_temperature"] = 15.0 * surface * profile
    dataset["sea_water_salinity"] = 35.0 * surface * profile
    matchups = _frame([("sea_surface_height_above_geoid", "surface")], start_dates, [0, 1], 20)
    document_rows = score_rows("rmsd_of_variables_compared_to_glorys_reanalysis", _gridded_metric(dataset, GLOBAL))

    viewer_data.build_viewer_data(str(tmp_path), "your_model", dataset, GLOBAL, matchups, document_rows)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "datasets.json",
        "insights",
        "insights.json",
        "scores-summary.json",
        "your_model.columns.zarr",
        "your_model.viewer-manifest.json",
        "your_model.zarr",
    ]
    manifest = json.loads((tmp_path / "your_model.viewer-manifest.json").read_text())
    assert manifest["has_columns"] is True
    assert set(manifest["provenance"]) == {"oceanbench_version", "generated_at"}
    insights = json.loads((tmp_path / "insights.json").read_text())["datasets"]["your_model"]["global"]
    assert insights["rmsd_by_depth"] is None
    for path in [path for path in insights.values() if path is not None]:
        assert (tmp_path / path.removeprefix("./data/")).is_file()
    summary = json.loads((tmp_path / "scores-summary.json").read_text())
    assert [(row["lead_day"], row["mean"], row["n_starts"]) for row in summary] == [(1, 1.0, 2), (2, 2.0, 2)]


def _summary_row(challenger: str, region: str) -> dict:
    return {
        "challenger": challenger,
        "year": 2024,
        "region": region,
        "metric": "rmsd",
        "reference": "glorys",
        "variable": "sea_water_salinity",
        "depth": "surface",
        "lead_day": 1,
    }


def test_indexes_merge_the_challenger_and_region_with_what_the_directory_holds(tmp_path) -> None:
    (tmp_path / "datasets.json").write_text(
        json.dumps({"datasets": [{"slug": "your_model", "label": "Your model"}, {"slug": "glo12", "label": "GLO12"}]})
    )
    (tmp_path / "scores-summary.json").write_text(
        json.dumps(
            [_summary_row("glo12", "global"), _summary_row("your_model", "ibi"), _summary_row("your_model", "global")]
        )
    )
    insights = {"class4_matchups": "./data/insights/your_model/global/class4-matchups.parquet"}

    viewer_data.update_viewer_indexes(
        tmp_path,
        "your_model",
        GLOBAL,
        ViewerArtifacts(insights, True),
        [_summary_row("your_model", "global") | {"lead_day": 2}],
    )

    datasets = json.loads((tmp_path / "datasets.json").read_text())["datasets"]
    assert datasets == [
        {"slug": "glo12", "label": "GLO12"},
        {
            "slug": "your_model",
            "label": "Your model",
            "store": "./data/your_model.zarr",
            "manifest": "./data/your_model.viewer-manifest.json",
            "columns": "./data/your_model.columns.zarr",
        },
    ]
    index = json.loads((tmp_path / "insights.json").read_text())
    assert index == {
        "datasets": {"your_model": {"global": insights}},
        "regions": {"global": None},
        "scores_summary": "./data/scores-summary.json",
    }
    summary = json.loads((tmp_path / "scores-summary.json").read_text())
    assert [(row["challenger"], row["region"], row["lead_day"]) for row in summary] == [
        ("glo12", "global", 1),
        ("your_model", "global", 2),
        ("your_model", "ibi", 1),
    ]


def test_evaluate_builds_the_viewer_data_from_the_class4_matchups_of_the_scores(
    evaluated_metrics, tmp_path, monkeypatch  # noqa: F811
) -> None:
    oceanbench.evaluate_challenger("challenger.py", region="ibi")
    scores_without_viewer_data = json.loads((tmp_path / "challenger.ibi.scores.json").read_text())
    matchups = pandas.DataFrame({"variable": ["sea_water_salinity"]})
    builds = []

    def class4_rmsd_and_matchups(challenger_dataset, region):
        return getattr(oceanbench.metrics, CLASS4_METRIC_FUNCTION_NAME)(challenger_dataset, region), matchups

    monkeypatch.setattr(
        evaluate.core_metrics, "class4_rmsd_and_matchups_compared_to_observations", class4_rmsd_and_matchups
    )
    monkeypatch.setattr(evaluate, "build_viewer_data", lambda *arguments: builds.append(arguments))

    oceanbench.evaluate_challenger("challenger.py", region="ibi", viewer_artifacts_directory=str(tmp_path / "viewer"))

    scores = json.loads((tmp_path / "challenger.ibi.scores.json").read_text())
    assert scores["rows"] == scores_without_viewer_data["rows"]
    ((directory, challenger_name, _, region, built_matchups, document_rows),) = builds
    assert (directory, challenger_name, region.id) == (str(tmp_path / "viewer"), "challenger", "ibi")
    assert built_matchups is matchups
    assert document_rows == scores["rows"]
    assert len(evaluated_metrics) == 2 * len(METRIC_FUNCTION_NAMES)


def test_evaluate_reads_the_viewer_data_directory_from_the_environment(
    evaluated_metrics, tmp_path, monkeypatch  # noqa: F811
) -> None:
    monkeypatch.setenv("OCEANBENCH_VIEWER_ARTIFACTS_DIR", str(tmp_path / "viewer"))
    monkeypatch.setattr(
        evaluate.core_metrics,
        "class4_rmsd_and_matchups_compared_to_observations",
        lambda challenger_dataset, region: (
            getattr(oceanbench.metrics, CLASS4_METRIC_FUNCTION_NAME)(challenger_dataset, region),
            None,
        ),
    )
    builds = []
    monkeypatch.setattr(evaluate, "build_viewer_data", lambda *arguments: builds.append(arguments))

    oceanbench.evaluate_challenger("challenger.py")

    assert [arguments[0] for arguments in builds] == [str(tmp_path / "viewer")]


@pytest.mark.parametrize("arguments, expected", [([], None), (["--viewer-artifacts-dir", "viewer"], "viewer")])
def test_the_evaluate_command_passes_the_viewer_data_directory(monkeypatch, arguments, expected) -> None:
    from oceanbench import cli

    calls = []
    monkeypatch.setattr(
        cli,
        "_evaluate_all",
        lambda challengers, *other_arguments: calls.append(other_arguments[-1])
        or [cli.EvaluationResult(challenger=challenger) for challenger in challengers],
    )
    args = cli._build_parser()[0].parse_args(["evaluate", "challenger.py", *arguments])

    assert cli._run_evaluate(args) == 0
    assert calls == [expected]
