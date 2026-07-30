# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path
import sys

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers import observation_refresh  # noqa: E402


class FakeResponse:
    def __init__(self, status_code: int, payload: object | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        return self._payload


def _clear_refresh_environment(monkeypatch) -> None:
    for name in [
        "EDITO_ACCESS_TOKEN",
        "EDITO_OFFLINE_TOKEN",
        "OBS_AWS_ACCESS_KEY_ID",
        "OBS_AWS_SECRET_ACCESS_KEY",
        "OBS_TARGET_S3_ENDPOINT",
        "OBS_AWS_DEFAULT_REGION",
        "COPERNICUSMARINE_SERVICE_USERNAME",
        "COPERNICUSMARINE_SERVICE_PASSWORD",
    ]:
        monkeypatch.delenv(name, raising=False)


def _set_direct_credentials(monkeypatch) -> None:
    monkeypatch.setenv("EDITO_ACCESS_TOKEN", "edito-access-token")
    monkeypatch.setenv("OBS_AWS_ACCESS_KEY_ID", "obs-access-key")
    monkeypatch.setenv("OBS_AWS_SECRET_ACCESS_KEY", "obs-secret-key")


def test_refresh_skips_without_credentials(monkeypatch, capsys) -> None:
    _clear_refresh_environment(monkeypatch)

    def unexpected_request(*_, **__) -> None:
        raise AssertionError("No request should be sent without credentials")

    monkeypatch.setattr(observation_refresh.requests, "get", unexpected_request)
    monkeypatch.setattr(observation_refresh.requests, "post", unexpected_request)
    monkeypatch.setattr(observation_refresh.requests, "put", unexpected_request)
    monkeypatch.setattr(observation_refresh.requests, "delete", unexpected_request)

    observation_refresh.maybe_launch_daily_observation_refresh()

    assert "Skipping daily observation data refresh" in capsys.readouterr().out


def test_refresh_does_not_fail_build_when_token_refresh_fails(monkeypatch, capsys) -> None:
    _clear_refresh_environment(monkeypatch)
    monkeypatch.setenv("EDITO_OFFLINE_TOKEN", "invalid-edito-offline-token")
    monkeypatch.setenv("OBS_AWS_ACCESS_KEY_ID", "obs-access-key")
    monkeypatch.setenv("OBS_AWS_SECRET_ACCESS_KEY", "obs-secret-key")

    def fake_post(*_, **__) -> FakeResponse:
        return FakeResponse(400)

    def unexpected_request(*_, **__) -> None:
        raise AssertionError("Datalab app endpoints should not be called when token refresh fails")

    monkeypatch.setattr(observation_refresh.requests, "post", fake_post)
    monkeypatch.setattr(observation_refresh.requests, "get", unexpected_request)
    monkeypatch.setattr(observation_refresh.requests, "put", unexpected_request)
    monkeypatch.setattr(observation_refresh.requests, "delete", unexpected_request)

    observation_refresh.maybe_launch_daily_observation_refresh()

    output = capsys.readouterr().out
    assert "Could not launch daily observation data refresh" in output
    assert "EDITO token refresh failed with status 400" in output


def test_refresh_skips_when_process_is_running(monkeypatch, capsys) -> None:
    _clear_refresh_environment(monkeypatch)
    _set_direct_credentials(monkeypatch)

    def fake_get(*_, **__) -> FakeResponse:
        return FakeResponse(
            200,
            {
                "services": [
                    {
                        "id": observation_refresh.DAILY_OBSERVATION_PROCESS_NAME,
                        "status": "deployed",
                        "tasks": [{"status": {"status": "Running"}}],
                    }
                ],
            },
        )

    def unexpected_request(*_, **__) -> None:
        raise AssertionError("Running refresh process should not be replaced")

    monkeypatch.setattr(observation_refresh.requests, "get", fake_get)
    monkeypatch.setattr(observation_refresh.requests, "put", unexpected_request)
    monkeypatch.setattr(observation_refresh.requests, "delete", unexpected_request)

    observation_refresh.maybe_launch_daily_observation_refresh()

    assert "already running" in capsys.readouterr().out


def test_refresh_launches_when_service_is_absent(monkeypatch, capsys) -> None:
    _clear_refresh_environment(monkeypatch)
    _set_direct_credentials(monkeypatch)
    launch_payloads = []

    def fake_get(*_, **__) -> FakeResponse:
        return FakeResponse(200, {"services": []})

    def fake_put(*_, json: dict[str, object], **__) -> FakeResponse:
        launch_payloads.append(json)
        return FakeResponse(201)

    def unexpected_delete(*_, **__) -> None:
        raise AssertionError("Missing refresh process should not be deleted before launch")

    monkeypatch.setattr(observation_refresh.requests, "get", fake_get)
    monkeypatch.setattr(observation_refresh.requests, "delete", unexpected_delete)
    monkeypatch.setattr(observation_refresh.requests, "put", fake_put)

    observation_refresh.maybe_launch_daily_observation_refresh()

    assert len(launch_payloads) == 1
    assert launch_payloads[0]["name"] == observation_refresh.DAILY_OBSERVATION_PROCESS_NAME
    assert "Launched daily observation data refresh process" in capsys.readouterr().out


def test_refresh_relaunches_terminal_process(monkeypatch, capsys) -> None:
    _clear_refresh_environment(monkeypatch)
    _set_direct_credentials(monkeypatch)
    deleted_params = []
    launch_payloads = []

    def fake_get(*_, **__) -> FakeResponse:
        return FakeResponse(
            200,
            {
                "services": [
                    {
                        "id": observation_refresh.DAILY_OBSERVATION_PROCESS_NAME,
                        "status": "deployed",
                        "tasks": [{"status": {"status": "Succeeded"}}],
                    }
                ],
            },
        )

    def fake_delete(*_, params: dict[str, str], **__) -> FakeResponse:
        deleted_params.append(params)
        return FakeResponse(204)

    def fake_put(*_, json: dict[str, object], **__) -> FakeResponse:
        launch_payloads.append(json)
        return FakeResponse(201)

    monkeypatch.setattr(observation_refresh.requests, "get", fake_get)
    monkeypatch.setattr(observation_refresh.requests, "delete", fake_delete)
    monkeypatch.setattr(observation_refresh.requests, "put", fake_put)

    observation_refresh.maybe_launch_daily_observation_refresh()

    assert deleted_params == [{"path": observation_refresh.DAILY_OBSERVATION_PROCESS_NAME}]
    assert len(launch_payloads) == 1
    payload = launch_payloads[0]
    assert payload["catalogId"] == "process-playground"
    assert payload["packageName"] == "daily-observation-data"
    assert payload["packageVersion"] == observation_refresh.DAILY_OBSERVATION_PROCESS_VERSION
    assert payload["packageVersion"] == "0.1.8"
    assert payload["name"] == observation_refresh.DAILY_OBSERVATION_PROCESS_NAME
    assert "s3" not in payload["options"]
    inputs = payload["options"]["inputs"]
    assert inputs["S3_OUTPUT_FOLDER"] == ("oceanbench-bucket/public/live_observations/{compact_date}.zarr")
    assert inputs["OBS_AVAILABILITY_MANIFEST_URL"] == ("oceanbench-bucket/public/live_observations/availability.json")
    assert inputs["OBS_PUBLIC_OBSERVATION_TEMPLATE"] == (
        "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/public/live_observations/{compact_date}.zarr"
    )
    assert inputs["OBS_TARGET_S3_ENDPOINT"] == "https://s3.waw3-1.cloudferro.com"
    assert inputs["AWS_ACCESS_KEY_ID"] == "obs-access-key"
    assert inputs["AWS_SECRET_ACCESS_KEY"] == "obs-secret-key"
    assert "Launched daily observation data refresh process" in capsys.readouterr().out


def test_refresh_relaunches_deployed_process_without_tasks(monkeypatch, capsys) -> None:
    _clear_refresh_environment(monkeypatch)
    _set_direct_credentials(monkeypatch)
    deleted_params = []
    launch_payloads = []

    def fake_get(*_, **__) -> FakeResponse:
        return FakeResponse(
            200,
            {
                "apps": [
                    {
                        "id": observation_refresh.DAILY_OBSERVATION_PROCESS_NAME,
                        "name": observation_refresh.DAILY_OBSERVATION_PROCESS_NAME,
                        "friendlyName": observation_refresh.DAILY_OBSERVATION_PROCESS_NAME,
                        "status": "deployed",
                        "tasks": [],
                    }
                ],
            },
        )

    def fake_delete(*_, params: dict[str, str], **__) -> FakeResponse:
        deleted_params.append(params)
        return FakeResponse(204)

    def fake_put(*_, json: dict[str, object], **__) -> FakeResponse:
        launch_payloads.append(json)
        return FakeResponse(201)

    monkeypatch.setattr(observation_refresh.requests, "get", fake_get)
    monkeypatch.setattr(observation_refresh.requests, "delete", fake_delete)
    monkeypatch.setattr(observation_refresh.requests, "put", fake_put)

    observation_refresh.maybe_launch_daily_observation_refresh()

    assert deleted_params == [{"path": observation_refresh.DAILY_OBSERVATION_PROCESS_NAME}]
    assert len(launch_payloads) == 1
    assert "Launched daily observation data refresh process" in capsys.readouterr().out
