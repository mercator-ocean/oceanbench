# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path

import numpy
import xarray

from oceanbench.core import nrt_validation
from oceanbench.core.dataset_utils import Dimension, Variable


def _complete_observation_dataset() -> xarray.Dataset:
    observation_dimension = "observations"
    variables = {
        Dimension.TIME.key(): (observation_dimension, numpy.array(["2026-05-22"], dtype="datetime64[ns]")),
        Dimension.DEPTH.key(): (observation_dimension, [0.0]),
        Dimension.LATITUDE.key(): (observation_dimension, [0.0]),
        Dimension.LONGITUDE.key(): (observation_dimension, [0.0]),
    }
    for variable in (
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
        Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
        Variable.SEA_WATER_SALINITY,
        Variable.EASTWARD_SEA_WATER_VELOCITY,
        Variable.NORTHWARD_SEA_WATER_VELOCITY,
    ):
        variables[variable.key()] = (observation_dimension, [1.0])
    return xarray.Dataset(variables)


def test_latest_complete_class4_observation_day_probes_backwards(monkeypatch) -> None:
    opened_paths = []

    def open_dataset(path, *_, **__):
        opened_paths.append(path)
        if "20260522" not in path:
            raise FileNotFoundError(path)
        return _complete_observation_dataset()

    monkeypatch.setattr(nrt_validation.xarray, "open_dataset", open_dataset)

    latest_day = nrt_validation.latest_complete_class4_observation_day(
        "file:///tmp/observations/{compact_date}.zarr",
        search_end_day="2026-05-23",
        max_lookback_days=3,
    )

    assert latest_day == "2026-05-22"
    assert nrt_validation.forecast_init_for_observation_cutoff(latest_day) == "2026-05-12"
    assert opened_paths == [
        "/tmp/observations/20260523.zarr",
        "/tmp/observations/20260522.zarr",
    ]


def test_class4_observation_day_requires_non_empty_current_values(monkeypatch) -> None:
    dataset = _complete_observation_dataset()
    dataset[Variable.EASTWARD_SEA_WATER_VELOCITY.key()] = (
        dataset[Variable.EASTWARD_SEA_WATER_VELOCITY.key()].dims,
        [numpy.nan],
    )
    monkeypatch.setattr(nrt_validation.xarray, "open_dataset", lambda *_, **__: dataset)

    assert not nrt_validation.class4_observation_day_is_complete(
        "2026-05-22",
        "file:///tmp/observations/{day}.zarr",
    )


def test_validate_nrt_forecast_writes_demo_manifest_and_runs_live_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    evaluate_calls = []
    cleanup_calls = []
    observation_checks = []
    manifest_path = tmp_path / "manifest.json"
    forecast_path = tmp_path / "forecast.zarr"
    observation_template = "file:///tmp/observations/{compact_date}.zarr"

    monkeypatch.setattr(
        nrt_validation,
        "latest_complete_class4_observation_day",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("Latest observation search should not be called")),
    )
    monkeypatch.setattr(
        nrt_validation,
        "class4_observation_day_is_complete",
        lambda day, template: observation_checks.append((day, template)) or True,
    )
    monkeypatch.setattr(
        nrt_validation,
        "request_octo_forecast_generation",
        lambda **_: {
            "forecast_url": str(forecast_path),
            "status": "ready",
            "process_package_name": "glonet-inference",
            "process_package_version": "0.0.7",
            "pending_reason": None,
            "gpu_capacity_available": True,
            "running_inference_processes": 2,
            "temporary": True,
        },
    )
    monkeypatch.setattr(nrt_validation, "wait_for_forecast_zarr_success", lambda *_, **__: True)

    def evaluate_live_challenger(**kwargs):
        evaluate_calls.append(kwargs)

    monkeypatch.setattr(nrt_validation, "evaluate_live_challenger", evaluate_live_challenger)
    monkeypatch.setattr(
        nrt_validation,
        "delete_forecast_zarr_store",
        lambda forecast_url: cleanup_calls.append(forecast_url) or "Deleted 42 forecast Zarr objects",
    )

    result, written_manifest = nrt_validation.validate_nrt_forecast(
        octo_script="/tmp/octo/orchestration_job.py",
        octo_forecast_output_prefix="public/octo/v0/oceanbench-nrt-ai-dev",
        observation_zarr_template=observation_template,
        forecast_init="2026-05-13",
        observation_cutoff="2026-05-23",
        forecast_ready_timeout_seconds=0,
        forecast_ready_poll_seconds=1,
        manifest_path=str(manifest_path),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = manifest["evaluations"][0]

    assert written_manifest == str(manifest_path)
    assert result.status == "Complete"
    assert result.forecast_init == "2026-05-13"
    assert result.observation_cutoff == "2026-05-23"
    assert result.forecast_temporary is True
    assert result.forecast_cleanup_status == "Deleted 42 forecast Zarr objects"
    assert result.demo is True
    assert result.initial_condition_provenance_validated is False
    assert result.octo_process_package_name == "glonet-inference"
    assert result.octo_generation_status == "ready"
    assert result.octo_pending_reason is None
    assert result.octo_gpu_capacity_available is True
    assert result.octo_running_inference_processes == 2
    assert evaluation["note"].startswith("Demonstration only")
    assert evaluation["forecast_temporary"] is True
    assert evaluation["forecast_cleanup_status"] == "Deleted 42 forecast Zarr objects"
    assert evaluation["octo_generation_status"] == "ready"
    assert evaluation["octo_gpu_capacity_available"] is True
    assert evaluate_calls[0]["output_notebook_file_name"] == "glonet.latest.global.report.ipynb"
    assert cleanup_calls == [str(forecast_path)]
    assert observation_checks == [("2026-05-23", observation_template)]


def test_nrt_zarr_template_supports_compact_date() -> None:
    assert (
        nrt_validation._format_zarr_template(
            "2026-05-23",
            "https://example.test/observations/{compact_date}.zarr",
        )
        == "https://example.test/observations/20260523.zarr"
    )


def test_validate_nrt_forecast_rejects_inconsistent_pinned_target() -> None:
    try:
        nrt_validation.validate_nrt_forecast(
            forecast_init="2026-05-14",
            observation_cutoff="2026-05-23",
            skip_forecast_generation=True,
        )
    except ValueError as error:
        assert "is not 10 days before" in str(error)
    else:
        raise AssertionError("Expected pinned target mismatch to fail")


def test_validate_nrt_forecast_requires_complete_pinned_target_pair() -> None:
    try:
        nrt_validation.validate_nrt_forecast(
            forecast_init="2026-05-13",
            skip_forecast_generation=True,
        )
    except ValueError as error:
        assert "--forecast-init and --observation-cutoff" in str(error)
    else:
        raise AssertionError("Expected one-sided pinned target to fail")


def test_validate_nrt_forecast_can_clean_external_temporary_forecast(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cleanup_calls = []
    manifest_path = tmp_path / "manifest.json"
    forecast_template = str(tmp_path / "external-{date}.zarr")

    monkeypatch.setattr(
        nrt_validation,
        "latest_complete_class4_observation_day",
        lambda *_, **__: "2026-05-23",
    )
    monkeypatch.setattr(
        nrt_validation,
        "request_octo_forecast_generation",
        lambda **_: (_ for _ in ()).throw(AssertionError("Octo should not be called")),
    )
    monkeypatch.setattr(nrt_validation, "wait_for_forecast_zarr_success", lambda *_, **__: True)
    monkeypatch.setattr(nrt_validation, "evaluate_live_challenger", lambda **_: None)
    monkeypatch.setattr(
        nrt_validation,
        "delete_forecast_zarr_store",
        lambda forecast_url: cleanup_calls.append(forecast_url) or "Deleted temporary forecast",
    )

    result, _ = nrt_validation.validate_nrt_forecast(
        forecast_zarr_template=forecast_template,
        skip_forecast_generation=True,
        forecast_temporary=True,
        forecast_ready_timeout_seconds=0,
        forecast_ready_poll_seconds=1,
        manifest_path=str(manifest_path),
    )

    forecast_url = str(tmp_path / "external-2026-05-13.zarr")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = manifest["evaluations"][0]

    assert result.status == "Complete"
    assert result.forecast_temporary is True
    assert result.forecast_cleanup_status == "Deleted temporary forecast"
    assert evaluation["forecast_temporary"] is True
    assert evaluation["forecast_cleanup_status"] == "Deleted temporary forecast"
    assert cleanup_calls == [forecast_url]


def test_write_nrt_manifest_merges_existing_challenger_rows(tmp_path: Path) -> None:
    manifest_path = tmp_path / "nrt-validation-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-06-02T00:00:00Z",
                "evaluations": [
                    {
                        "system_id": "octo-glonet-p1d",
                        "system_label": "GLONET",
                        "region": "global",
                        "forecast_init": "2026-05-12",
                        "status": "Old",
                    },
                    {
                        "system_id": "other-system",
                        "system_label": "Other",
                        "region": "global",
                        "forecast_init": "2026-05-12",
                        "status": "Complete",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    new_manifest = {
        "schema_version": 1,
        "generated_at": "2026-06-03T00:00:00Z",
        "evaluations": [
            {
                "system_id": "octo-glonet-p1d",
                "system_label": "GLONET",
                "region": "global",
                "forecast_init": "2026-05-13",
                "status": "Complete",
            }
        ],
    }

    written_manifest = nrt_validation.write_nrt_manifest(
        new_manifest,
        manifest_path=str(manifest_path),
        output_bucket=None,
        output_prefix=None,
    )

    merged_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert written_manifest == str(manifest_path)
    assert merged_manifest["generated_at"] == "2026-06-03T00:00:00Z"
    assert merged_manifest["evaluations"] == [
        {
            "system_id": "octo-glonet-p1d",
            "system_label": "GLONET",
            "region": "global",
            "forecast_init": "2026-05-13",
            "status": "Complete",
        },
        {
            "system_id": "other-system",
            "system_label": "Other",
            "region": "global",
            "forecast_init": "2026-05-12",
            "status": "Complete",
        },
    ]


def test_request_octo_forecast_generation_passes_forecast_output_prefix(monkeypatch) -> None:
    commands = []

    class CompletedProcess:
        stdout = '{"forecast_url": "https://example.test/forecast.zarr"}'

    def run(command, **kwargs):
        commands.append((command, kwargs))
        return CompletedProcess()

    monkeypatch.setattr(nrt_validation.subprocess, "run", run)

    result = nrt_validation.request_octo_forecast_generation(
        octo_script="/tmp/octo/orchestration_job.py",
        system_id="octo-glonet-p1d",
        forecast_init="2026-05-13",
        python_executable="/tmp/octo/python",
        forecast_output_prefix="public/octo/v0/oceanbench-nrt-ai-dev",
    )

    assert result["forecast_url"] == "https://example.test/forecast.zarr"
    assert commands[0][0] == [
        "/tmp/octo/python",
        "/tmp/octo/orchestration_job.py",
        "generate-forecast",
        "--system-id",
        "octo-glonet-p1d",
        "--forecast-init",
        "2026-05-13",
        "--forecast-output-prefix",
        "public/octo/v0/oceanbench-nrt-ai-dev",
    ]
    assert commands[0][1]["check"] is True
