# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import inspect

import numpy
import pytest

import oceanbench.cli as cli
import oceanbench.core.metrics as core_metrics
import oceanbench.core.runtime_configuration as runtime_configuration
import oceanbench.core.stage as stage
from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.regions import resolve_region
from oceanbench.core.resolution import get_dataset_resolution
from oceanbench.core.runtime_configuration import STAGE_ALL_KEY, current_runtime_configuration

CHALLENGER_CODE = """
import numpy
import xarray

from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.regions import resolve_region

challenger_dataset = xarray.Dataset(
    coords={
        Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]"),
        Dimension.LEAD_DAY_INDEX.key(): numpy.arange(10),
        Dimension.DEPTH.key(): numpy.array([0.494025, 47.37369]),
        Dimension.LATITUDE.key(): numpy.array([0.0, 0.25, 0.5]),
        Dimension.LONGITUDE.key(): numpy.array([10.0, 10.25, 10.5]),
    }
)
"""

REFERENCE_OPENER_NAMES = ["glorys_reanalysis_dataset", "glo12_analysis_dataset", "observations"]


def _asset_key(opener_name: str, challenger_dataset) -> tuple:
    return (
        opener_name,
        tuple(challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values.tolist()),
        challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()],
        tuple(challenger_dataset[Dimension.DEPTH.key()].values.tolist()),
        get_dataset_resolution(challenger_dataset),
    )


def _record_asset_openers(monkeypatch, module, opened_assets: set) -> None:
    for opener_name in REFERENCE_OPENER_NAMES:

        def record_reference(challenger_dataset, opener_name=opener_name):
            assert current_runtime_configuration().staged_components == (STAGE_ALL_KEY,)
            opened_assets.add(_asset_key(opener_name, challenger_dataset))
            return challenger_dataset

        monkeypatch.setattr(module, opener_name, record_reference)
    monkeypatch.setattr(module, "ocean_mask", lambda: opened_assets.add(("ocean_mask",)))


def _forbid(*_args, **_kwargs):
    raise AssertionError("Staging must not compute any score or delete the stage directory.")


@pytest.fixture
def challenger_file(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_configuration, "_runtime_configuration", None)
    path = tmp_path / "challenger.py"
    path.write_text(CHALLENGER_CODE, encoding="utf8")
    return path


def test_stage_command_stages_every_asset_and_computes_nothing(challenger_file, tmp_path, monkeypatch) -> None:
    staged_assets: set = set()
    _record_asset_openers(monkeypatch, stage, staged_assets)
    monkeypatch.setattr(
        stage,
        "load_mean_dynamic_topography",
        lambda resolution: staged_assets.add(("mean_dynamic_topography", resolution)),
    )
    for metric_computation in ("rmsd", "rmsd_class4_validation", "deviation_of_lagrangian_trajectories"):
        monkeypatch.setattr(core_metrics, metric_computation, _forbid)
    monkeypatch.setattr(cli, "cleanup_local_stage_directory", _forbid)
    stage_directory = tmp_path / "stage"
    args = cli._build_parser()[0].parse_args(["stage", str(challenger_file), "--stage-dir", str(stage_directory)])

    assert cli._run_stage(args) == 0

    first_day_datetimes = tuple(numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]").tolist())
    reference_key = (first_day_datetimes, 10, (0.494025, 47.37369), "quarter_degree")
    assert staged_assets == {
        ("glorys_reanalysis_dataset", *reference_key),
        ("glo12_analysis_dataset", *reference_key),
        ("observations", *reference_key),
        ("mean_dynamic_topography", "quarter_degree"),
        ("ocean_mask",),
    }
    assert current_runtime_configuration().resolved_stage_directory() == stage_directory


def test_stage_command_opens_the_same_assets_as_the_metrics(challenger_file, monkeypatch) -> None:
    staged_assets: set = set()
    _record_asset_openers(monkeypatch, stage, staged_assets)
    monkeypatch.setattr(stage, "load_mean_dynamic_topography", lambda resolution: None)
    stage.stage_challenger_assets(
        str(challenger_file),
        runtime_configuration.RuntimeConfiguration(staged_components=(STAGE_ALL_KEY,)),
    )

    metric_assets: set = set()
    _record_asset_openers(monkeypatch, core_metrics, metric_assets)
    for computation in (
        "rmsd",
        "rmsd_class4_validation",
        "deviation_of_lagrangian_trajectories",
        "compute_mixed_layer_depth",
        "compute_geostrophic_currents",
        "_lagrangian_particle_count",
    ):
        monkeypatch.setattr(core_metrics, computation, lambda *args, **kwargs: None)
    challenger_namespace: dict = {}
    exec(CHALLENGER_CODE, challenger_namespace)
    metric_functions = [
        function
        for name, function in inspect.getmembers(core_metrics, inspect.isfunction)
        if function.__module__ == core_metrics.__name__ and not name.startswith("_")
    ]
    for metric_function in metric_functions:
        metric_function(challenger_namespace["challenger_dataset"])

    assert len(metric_functions) == 9
    assert metric_assets == staged_assets


def test_stage_command_reads_the_region_from_the_environment_like_evaluate(challenger_file, monkeypatch) -> None:
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_REGION.value, "ibi")
    _record_asset_openers(monkeypatch, stage, set())
    monkeypatch.setattr(stage, "load_mean_dynamic_topography", lambda resolution: None)
    subset_regions = []
    monkeypatch.setattr(
        stage,
        "subset_dataset_to_region",
        lambda dataset, region: subset_regions.append(region) or dataset,
    )

    stage.stage_challenger_assets(
        str(challenger_file),
        runtime_configuration.RuntimeConfiguration(staged_components=(STAGE_ALL_KEY,)),
    )

    assert subset_regions == [resolve_region("ibi")]


def _run_successful_evaluate(monkeypatch, arguments: list[str]) -> list:
    for environment_variable in (
        OceanbenchEnvironmentVariable.OCEANBENCH_STAGE,
        OceanbenchEnvironmentVariable.OCEANBENCH_STAGE_DIR,
    ):
        monkeypatch.delenv(environment_variable.value, raising=False)
    monkeypatch.setattr(
        cli,
        "_evaluate_all",
        lambda challengers, *_args: [cli.EvaluationResult(challenger=challenger) for challenger in challengers],
    )
    deleted_directories = []
    monkeypatch.setattr(cli, "cleanup_local_stage_directory", deleted_directories.append)
    args = cli._build_parser()[0].parse_args(["evaluate", "challenger.py", *arguments])

    assert cli._run_evaluate(args) == 0
    return deleted_directories


def test_evaluate_keeps_a_named_stage_directory(tmp_path, monkeypatch) -> None:
    assert _run_successful_evaluate(monkeypatch, ["--stage-dir", str(tmp_path / "stage")]) == []


def test_evaluate_deletes_the_default_stage_directory(monkeypatch) -> None:
    deleted_directories = _run_successful_evaluate(monkeypatch, ["--stage", STAGE_ALL_KEY])

    assert deleted_directories == [runtime_configuration.RuntimeConfiguration().resolved_stage_directory()]


def test_evaluate_keep_stage_keeps_the_default_stage_directory(monkeypatch) -> None:
    assert _run_successful_evaluate(monkeypatch, ["--stage", STAGE_ALL_KEY, "--keep-stage"]) == []
