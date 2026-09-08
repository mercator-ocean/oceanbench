# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Unit tests for the S3 publish step. No network access is performed."""

from unittest import mock

import pytest

from oceanbench.publish import s3


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("viewer/index.html", "text/html"),
        ("viewer/app.js", "text/javascript"),
        ("viewer/style.css", "text/css"),
        ("catalog.json", "application/json"),
        ("viewer/icon.svg", "image/svg+xml"),
        ("viewer/preview.png", "image/png"),
        ("favicon.ico", "image/x-icon"),
        ("README.md", "text/markdown"),
        ("viewer/verify_viewer.mjs", "text/javascript"),
        ("scores.parquet", "application/vnd.apache.parquet"),
    ],
)
def test_content_type_maps_known_suffixes(path, content_type):
    assert s3.content_type_for_path(path) == content_type


@pytest.mark.parametrize(
    "path",
    [
        "viewer/2024/glonet.zarr/temperature/0.0.0",
        "viewer/2024/glonet.zarr/.zattrs",
        "viewer/2024/glonet.zarr/.zmetadata",
        "viewer/2024/glonet.zarr/temperature/0",
        "viewer/2024/glonet.zarr/temperature/chunk",
    ],
)
def test_content_type_keeps_zarr_chunks_and_metadata_as_octet_stream(path):
    assert s3.content_type_for_path(path) == "application/octet-stream"


def _write(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_build_upload_plan_preserves_layout_and_prefix(tmp_path):
    root = tmp_path / "tree"
    _write(root / "catalog.json", b"{}")
    _write(root / "scores.parquet", b"abcd")
    _write(root / "2024" / "ibi" / "glonet" / "insights" / "manifest.json", b"{}")

    plan = s3.build_upload_plan(root, "dev/benchmark/rebuild-preview")

    keys = [item.key for item in plan]
    assert keys == [
        "dev/benchmark/rebuild-preview/2024/ibi/glonet/insights/manifest.json",
        "dev/benchmark/rebuild-preview/catalog.json",
        "dev/benchmark/rebuild-preview/scores.parquet",
    ]
    assert {item.size for item in plan} == {2, 4}


def test_build_upload_plan_strips_prefix_slashes(tmp_path):
    _write(tmp_path / "a.json")
    plan = s3.build_upload_plan(tmp_path, "/dev/benchmark/x/")
    assert plan[0].key == "dev/benchmark/x/a.json"


def test_build_upload_plan_rejects_missing_root(tmp_path):
    with pytest.raises(NotADirectoryError):
        s3.build_upload_plan(tmp_path / "does-not-exist", "p")


def test_build_upload_plan_follows_symlinked_directories(tmp_path):
    # The viewer pyramid dirs are symlinks; their files must be published, not skipped.
    real_pyramid = tmp_path / "real_pyramid"
    _write(real_pyramid / "temperature" / "0.0.0", b"chunk")
    _write(real_pyramid / ".zattrs", b"{}")

    tree = tmp_path / "tree"
    _write(tree / "catalog.json", b"{}")
    (tree / "viewer" / "glonet.zarr").parent.mkdir(parents=True, exist_ok=True)
    (tree / "viewer" / "glonet.zarr").symlink_to(real_pyramid, target_is_directory=True)

    plan = s3.build_upload_plan(tree, "dev/x")

    keys = sorted(item.key for item in plan)
    assert keys == [
        "dev/x/catalog.json",
        "dev/x/viewer/glonet.zarr/.zattrs",
        "dev/x/viewer/glonet.zarr/temperature/0.0.0",
    ]


def test_build_upload_plan_survives_symlink_cycle(tmp_path):
    tree = tmp_path / "tree"
    _write(tree / "a.json", b"{}")
    # A directory symlink pointing back at the tree root would loop forever without a guard.
    (tree / "loop").symlink_to(tree, target_is_directory=True)

    plan = s3.build_upload_plan(tree, "p")

    assert any(item.key == "p/a.json" for item in plan)
    assert all("does-not-exist" not in item.key for item in plan)


def _plan_item(size=10, key="p/a.json"):
    return s3.UploadPlanItem(local_path="a.json", key=key, size=size)


def test_should_skip_when_remote_size_matches():
    client = mock.Mock()
    client.head_object.return_value = {"ContentLength": 10}
    assert s3.should_skip_upload(client, "bucket", _plan_item(size=10), force=False) is True


def test_should_not_skip_when_remote_size_differs():
    client = mock.Mock()
    client.head_object.return_value = {"ContentLength": 11}
    assert s3.should_skip_upload(client, "bucket", _plan_item(size=10), force=False) is False


def test_should_not_skip_when_remote_missing():
    from botocore.exceptions import ClientError

    client = mock.Mock()
    client.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")
    assert s3.should_skip_upload(client, "bucket", _plan_item(), force=False) is False


def test_force_never_skips_and_never_hits_the_network():
    client = mock.Mock()
    assert s3.should_skip_upload(client, "bucket", _plan_item(), force=True) is False
    client.head_object.assert_not_called()


def test_resolve_credentials_reads_the_aws_environment_variables():
    environment = {
        "AWS_ACCESS_KEY_ID": "AKIA",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_SESSION_TOKEN": "token",
    }
    credentials = s3.resolve_credentials(environment=environment)
    assert credentials.source == "aws-env"
    assert credentials.access_key_id == "AKIA"
    assert credentials.session_token == "token"


def test_resolve_credentials_raises_when_nothing_available():
    with pytest.raises(RuntimeError, match="AWS_ACCESS_KEY_ID"):
        s3.resolve_credentials(environment={})


def test_default_endpoint_is_cloudferro_unless_overridden():
    assert s3.default_endpoint(environment={}) == s3.CLOUDFERRO_ENDPOINT
    assert s3.default_endpoint(environment={"AWS_S3_ENDPOINT": "https://example.org"}) == "https://example.org"


def test_boto3_client_keyword_arguments_omits_absent_session_token():
    without_token = s3.AwsCredentials("k", "s", None, source="aws-env")
    assert "aws_session_token" not in without_token.boto3_client_keyword_arguments()
    with_token = s3.AwsCredentials("k", "s", "t", source="aws-env")
    assert with_token.boto3_client_keyword_arguments()["aws_session_token"] == "t"


def test_upload_one_gzips_json_with_content_encoding(tmp_path):
    import gzip

    json_path = tmp_path / "eddies.json"
    payload = b'{"kind": "eddy-census", "frames": []}'
    json_path.write_bytes(payload)
    client = mock.Mock()
    client.head_object.side_effect = _missing_object()
    item = s3.UploadPlanItem(local_path=json_path, key="p/eddies.json", size=len(payload))

    stored_item, was_uploaded = s3._upload_one(client, "bucket", item, force=False, compress_json=True)

    assert was_uploaded is True
    call = client.put_object.call_args.kwargs
    assert call["ContentType"] == "application/json"
    assert call["ContentEncoding"] == "gzip"
    assert gzip.decompress(call["Body"]) == payload
    assert stored_item.size == len(call["Body"])
    client.upload_file.assert_not_called()


def test_upload_one_leaves_non_json_untouched(tmp_path):
    parquet_path = tmp_path / "scores.parquet"
    parquet_path.write_bytes(b"PAR1")
    client = mock.Mock()
    client.head_object.side_effect = _missing_object()
    item = s3.UploadPlanItem(local_path=parquet_path, key="p/scores.parquet", size=4)

    s3._upload_one(client, "bucket", item, force=False, compress_json=True)

    client.put_object.assert_not_called()
    client.upload_file.assert_called_once()


def _missing_object():
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": "404"}}, "HeadObject")
