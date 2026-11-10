# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
import sys
from pathlib import Path

EDITO_DIRECTORY = Path(__file__).resolve().parents[1] / "edito"
sys.path.insert(0, str(EDITO_DIRECTORY))

from release_candidate import (  # noqa: E402
    FRAGMENTS_DIRECTORY,
    FRIENDLY_NAME_MAXIMUM_LENGTH,
    Job,
    Release,
    ResourceClass,
    RunMode,
    assemble,
    credentials_from_environment,
    official_challengers,
    process_payload,
    release_jobs,
    resource_class,
    viewer_file_destination,
)

RELEASE = Release(version="0.7.0rc1", git_reference="release-candidate-pipeline")
ENVIRONMENT = {
    "AWS_S3_ENDPOINT": "s3.example.org",
    "AWS_DEFAULT_REGION": "waw3-1",
    "AWS_ACCESS_KEY_ID": "key",
    "AWS_SECRET_ACCESS_KEY": "secret",
    "COPERNICUSMARINE_SERVICE_USERNAME": "user",
    "COPERNICUSMARINE_SERVICE_PASSWORD": "password",
}


def test_every_official_challenger_is_evaluated_on_the_global_region_and_only_high_resolution_ones_on_ibi():
    jobs = {job.name() for job in release_jobs(official_challengers())}

    assert "glonet.global" in jobs
    assert "glonet.ibi" in jobs
    assert "glonet_1_degree.global" in jobs
    assert "glonet_1_degree.ibi" not in jobs
    assert "hclimrep.ibi" not in jobs
    assert {job.name() for job in release_jobs(["xihe"])} == {"xihe.global", "xihe.ibi"}


def test_resources_follow_the_grid_of_the_challenger():
    assert resource_class("glonet_1_degree") is ResourceClass.ONE_DEGREE
    assert resource_class("xihe") is ResourceClass.QUARTER_DEGREE
    assert resource_class("glo12") is ResourceClass.TWELFTH_DEGREE


def test_the_process_runs_the_release_candidate_code_for_one_challenger_and_region():
    payload = process_payload(
        RELEASE, Job("xihe", "ibi"), credentials_from_environment(ENVIRONMENT, RunMode.FULL), RunMode.FULL
    )
    inputs = payload["processInputs"]
    variables = {variable["name"]: variable["value"] for variable in inputs["processInputs"]}

    assert inputs["configuration"]["git"]["branch"] == "release-candidate-pipeline"
    assert variables["CHALLENGER"] == "xihe"
    assert variables["REGION"] == "ibi"
    assert variables["REPORTS_PREFIX"] == "public/evaluation-reports/0.7.0rc1"
    assert variables["VIEWER_PREFIX"] == "viewer/data-0.7.0rc1"
    assert "DRY_RUN" not in variables
    assert inputs["s3"]["accessKeyId"] == "key"
    assert inputs["copernicusMarine"]["username"] == "user"
    assert len(inputs["onyxia"]["friendlyName"]) <= FRIENDLY_NAME_MAXIMUM_LENGTH


def test_a_dry_run_carries_no_credential_and_evaluates_nothing():
    payload = process_payload(RELEASE, Job("xihe", "ibi"), credentials_from_environment({}, RunMode.DRY), RunMode.DRY)
    inputs = payload["processInputs"]

    assert inputs["s3"]["enabled"] is False
    assert inputs["copernicusMarine"]["enabled"] is False
    assert {"name": "DRY_RUN", "value": "1"} in inputs["processInputs"]


def test_index_files_are_uploaded_as_fragments_of_their_job_and_other_files_in_place():
    job = Job("xihe", "ibi")
    viewer_prefix = RELEASE.viewer_prefix()

    assert (
        viewer_file_destination(viewer_prefix, job, Path("insights.json"))
        == f"{viewer_prefix}/{FRAGMENTS_DIRECTORY}/xihe.ibi/insights.json"
    )
    assert (
        viewer_file_destination(viewer_prefix, job, Path("insights/xihe/ibi/insights.json"))
        == f"{viewer_prefix}/insights/xihe/ibi/insights.json"
    )
    assert viewer_file_destination(viewer_prefix, job, Path("xihe.zarr/.zmetadata")) == f"{viewer_prefix}/xihe.zarr/.zmetadata"


class FakeBody:
    def __init__(self, content: bytes):
        self.content = content

    def read(self) -> bytes:
        return self.content


class FakeStorage:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = dict(objects)

    def get_paginator(self, name: str):
        storage = self

        class Paginator:
            def paginate(self, Bucket: str, Prefix: str):
                return [{"Contents": [{"Key": key} for key in storage.objects if key.startswith(Prefix)]}]

        return Paginator()

    def get_object(self, Bucket: str, Key: str) -> dict:
        return {"Body": FakeBody(self.objects[Key])}

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:
        self.objects[Key] = Body


def _fragment_objects(challenger: str, region: str) -> dict[str, bytes]:
    prefix = f"{RELEASE.viewer_prefix()}/{FRAGMENTS_DIRECTORY}/{challenger}.{region}"
    row = {
        "challenger": challenger,
        "year": 2024,
        "region": region,
        "metric": "rmsd_of_variables",
        "reference": "glorys",
        "variable": "temperature",
        "depth": "surface",
        "lead_day": 1,
    }
    return {
        f"{prefix}/datasets.json": json.dumps({"datasets": [{"slug": challenger}]}).encode(),
        f"{prefix}/insights.json": json.dumps(
            {"datasets": {challenger: {region: {}}}, "regions": {region: None}, "scores_summary": "./data/s.json"}
        ).encode(),
        f"{prefix}/scores-summary.json": json.dumps([row]).encode(),
    }


def test_assembling_merges_the_fragments_of_every_job_into_the_viewer_index_files():
    storage = FakeStorage({**_fragment_objects("glonet", "global"), **_fragment_objects("xihe", "ibi")})

    counts = assemble(storage, "bucket", RELEASE.viewer_prefix())

    prefix = RELEASE.viewer_prefix()
    assert counts == {"datasets.json": 2, "insights.json": 2, "scores-summary.json": 2}
    assert [entry["slug"] for entry in json.loads(storage.objects[f"{prefix}/datasets.json"])["datasets"]] == [
        "glonet",
        "xihe",
    ]
    assert sorted(json.loads(storage.objects[f"{prefix}/insights.json"])["datasets"]) == ["glonet", "xihe"]
    assert len(json.loads(storage.objects[f"{prefix}/scores-summary.json"])) == 2
