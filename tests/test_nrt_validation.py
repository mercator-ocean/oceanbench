# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _score_table() -> str:
    return (
        "<table>"
        "<thead><tr><th></th><th>Lead day 1</th><th>Lead day 10</th></tr></thead>"
        "<tbody>"
        "<tr><th>Temperature (C) [sea_water_potential_temperature]{surface}</th><td>1.2</td><td>1.4</td></tr>"
        "<tr><th>Salinity (PSU) [sea_water_salinity]{0-5m}</th><td>0.3</td><td>0.4</td></tr>"
        "<tr><th>Zonal current (m/s) [eastward_sea_water_velocity]{15m}</th><td>0.2</td><td>0.25</td></tr>"
        "<tr><th>Meridional current (m/s) [northward_sea_water_velocity]{15m}</th><td>0.21</td><td>0.24</td></tr>"
        "</tbody>"
        "</table>"
    )


def _write_report_notebook(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": "evaluation_report.class4_observation.rmsd",
                        "outputs": [{"data": {"text/html": _score_table()}}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


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


def test_validate_nrt_forecast_writes_manifest_and_runs_live_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    evaluate_calls = []
    cleanup_calls = []
    observation_checks = []
    manifest_path = tmp_path / "manifest.json"
    forecast_path = tmp_path / "forecast.zarr"
    observation_template = "file:///tmp/observations/{compact_date}.zarr"
    report_directory = tmp_path / "reports"

    monkeypatch.setattr(
        nrt_validation,
        "class4_observation_day_is_complete",
        lambda day, template: observation_checks.append((day, template)) or True,
    )
    monkeypatch.setattr(nrt_validation, "wait_for_forecast_zarr_success", lambda *_, **__: True)
    monkeypatch.setattr(nrt_validation, "_forecast_lead_day_count", lambda *_, **__: 10)

    def evaluate_live_challenger(**kwargs):
        evaluate_calls.append(kwargs)
        _write_report_notebook(Path(kwargs["output_prefix"]) / kwargs["output_notebook_file_name"])

    monkeypatch.setattr(nrt_validation, "evaluate_live_challenger", evaluate_live_challenger)
    monkeypatch.setattr(
        nrt_validation,
        "delete_forecast_zarr_store",
        lambda forecast_url: cleanup_calls.append(forecast_url) or "Deleted 42 forecast Zarr objects",
    )

    result, written_manifest = nrt_validation.validate_nrt_forecast(
        forecast_zarr_template=str(forecast_path),
        observation_zarr_template=observation_template,
        forecast_init="2026-05-13",
        observation_cutoff="2026-05-23",
        forecast_temporary=True,
        forecast_ready_timeout_seconds=0,
        forecast_ready_poll_seconds=1,
        manifest_path=str(manifest_path),
        output_prefix=str(report_directory),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = manifest["evaluations"][0]

    assert written_manifest == str(manifest_path)
    assert result.status == "Complete"
    assert result.forecast_init == "2026-05-13"
    assert result.forecast_lead_days == 10
    assert result.validated_lead_days == "1-10 days"
    assert result.observation_cutoff == "2026-05-23"
    assert result.forecast_temporary is True
    assert result.forecast_cleanup_status == "Deleted 42 forecast Zarr objects"
    assert "demo" not in evaluation
    assert "initial_condition_provenance_validated" not in evaluation
    assert "note" not in evaluation
    assert evaluation["forecast_temporary"] is True
    assert evaluation["forecast_cleanup_status"] == "Deleted 42 forecast Zarr objects"
    assert evaluation["score_preview"] == {
        "metrics": [
            {
                "label": "Temperature, surface",
                "unit": "C",
                "lead_values": {"1": 1.2, "10": 1.4},
            },
            {
                "label": "Salinity, 0-5 m",
                "unit": "PSU",
                "lead_values": {"1": 0.3, "10": 0.4},
            },
            {
                "label": "Zonal current, 15 m",
                "unit": "m/s",
                "lead_values": {"1": 0.2, "10": 0.25},
            },
            {
                "label": "Meridional current, 15 m",
                "unit": "m/s",
                "lead_values": {"1": 0.21, "10": 0.24},
            },
        ]
    }
    assert evaluate_calls[0]["output_notebook_file_name"] == "glonet.latest.global.report.ipynb"
    assert evaluate_calls[0]["output_prefix"] == str(report_directory)
    assert cleanup_calls == [str(forecast_path)]
    assert observation_checks == [("2026-05-23", observation_template)]


def test_score_preview_from_rmsd_html_uses_representative_metrics() -> None:
    assert nrt_validation._score_preview_from_rmsd_html(_score_table()) == {
        "metrics": [
            {
                "label": "Temperature, surface",
                "unit": "C",
                "lead_values": {"1": 1.2, "10": 1.4},
            },
            {
                "label": "Salinity, 0-5 m",
                "unit": "PSU",
                "lead_values": {"1": 0.3, "10": 0.4},
            },
            {
                "label": "Zonal current, 15 m",
                "unit": "m/s",
                "lead_values": {"1": 0.2, "10": 0.25},
            },
            {
                "label": "Meridional current, 15 m",
                "unit": "m/s",
                "lead_values": {"1": 0.21, "10": 0.24},
            },
        ]
    }


def test_validate_nrt_forecast_records_actual_forecast_lead_count(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    forecast_path = tmp_path / "langya.zarr"

    monkeypatch.setattr(nrt_validation, "class4_observation_day_is_complete", lambda *_, **__: True)
    monkeypatch.setattr(nrt_validation, "wait_for_forecast_zarr_success", lambda *_, **__: True)
    monkeypatch.setattr(nrt_validation, "_forecast_lead_day_count", lambda *_, **__: 7)
    monkeypatch.setattr(nrt_validation, "evaluate_live_challenger", lambda **_: None)

    result, _ = nrt_validation.validate_nrt_forecast(
        system_label="LangYa",
        forecast_zarr_template=str(forecast_path),
        forecast_init="2026-05-13",
        observation_cutoff="2026-05-23",
        forecast_ready_timeout_seconds=0,
        forecast_ready_poll_seconds=1,
        manifest_path=str(manifest_path),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = manifest["evaluations"][0]

    assert result.forecast_lead_days == 7
    assert result.validated_lead_days == "1-7 days"
    assert evaluation["forecast_lead_days"] == 7
    assert evaluation["validated_lead_days"] == "1-7 days"


def test_nrt_zarr_template_supports_compact_date() -> None:
    assert (
        nrt_validation._format_zarr_template(
            "2026-05-23",
            "https://example.test/observations/{compact_date}.zarr",
        )
        == "https://example.test/observations/20260523.zarr"
    )


def test_delete_s3_prefix_deletes_objects_individually(monkeypatch) -> None:
    deleted_objects = []

    class FakePaginator:
        def paginate(self, **kwargs):
            assert kwargs == {
                "Bucket": "bucket",
                "Prefix": "forecast.zarr/",
            }
            return [
                {
                    "Contents": [
                        {"Key": "forecast.zarr/.zgroup"},
                        {"Key": "forecast.zarr/_SUCCESS"},
                    ]
                }
            ]

    class FakeClient:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return FakePaginator()

        def delete_object(self, **kwargs):
            deleted_objects.append(kwargs)

        def delete_objects(self, **kwargs):
            raise AssertionError("Bulk delete should not be used")

    fake_boto3 = SimpleNamespace(client=lambda *_, **__: FakeClient())
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    assert nrt_validation._delete_s3_prefix("bucket", "forecast.zarr", None) == 2
    assert deleted_objects == [
        {"Bucket": "bucket", "Key": "forecast.zarr/.zgroup"},
        {"Bucket": "bucket", "Key": "forecast.zarr/_SUCCESS"},
    ]


def test_validate_nrt_forecast_rejects_inconsistent_pinned_target() -> None:
    try:
        nrt_validation.validate_nrt_forecast(
            forecast_init="2026-05-14",
            observation_cutoff="2026-05-23",
        )
    except ValueError as error:
        assert "is not 10 days before" in str(error)
    else:
        raise AssertionError("Expected pinned target mismatch to fail")


def test_validate_nrt_forecast_requires_complete_pinned_target_pair() -> None:
    try:
        nrt_validation.validate_nrt_forecast(
            forecast_init="2026-05-13",
        )
    except ValueError as error:
        assert "--forecast-init and --observation-cutoff are required" in str(error)
    else:
        raise AssertionError("Expected one-sided pinned target to fail")


def test_validate_nrt_forecast_can_clean_external_temporary_forecast(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cleanup_calls = []
    manifest_path = tmp_path / "manifest.json"
    forecast_template = str(tmp_path / "external-{date}.zarr")

    monkeypatch.setattr(nrt_validation, "class4_observation_day_is_complete", lambda *_, **__: True)
    monkeypatch.setattr(nrt_validation, "wait_for_forecast_zarr_success", lambda *_, **__: True)
    monkeypatch.setattr(nrt_validation, "_forecast_lead_day_count", lambda *_, **__: 10)
    monkeypatch.setattr(nrt_validation, "evaluate_live_challenger", lambda **_: None)
    monkeypatch.setattr(
        nrt_validation,
        "delete_forecast_zarr_store",
        lambda forecast_url: cleanup_calls.append(forecast_url) or "Deleted temporary forecast",
    )

    result, _ = nrt_validation.validate_nrt_forecast(
        forecast_zarr_template=forecast_template,
        forecast_init="2026-05-13",
        observation_cutoff="2026-05-23",
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
