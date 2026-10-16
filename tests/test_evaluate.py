# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
import sys

import pandas
import pytest

import oceanbench
import oceanbench.core.evaluate as evaluate
import oceanbench.core.runtime_configuration as runtime_configuration
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.runtime_configuration import RuntimeConfiguration, current_runtime_configuration
from oceanbench.core.scores import METRIC_FUNCTION_NAMES

CHALLENGER_CODE = """
import numpy
import xarray

challenger_dataset = xarray.Dataset(
    coords={
        "first_day_datetime": numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]"),
        "lead_day_index": numpy.arange(2),
    }
)
"""


@pytest.fixture
def evaluated_metrics(tmp_path, monkeypatch) -> list:
    monkeypatch.setattr(runtime_configuration, "_runtime_configuration", None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "challenger.py").write_text(CHALLENGER_CODE, encoding="utf8")
    calls = []
    for metric_function_name in METRIC_FUNCTION_NAMES:

        def metric(challenger_dataset, region, metric_function_name=metric_function_name):
            calls.append((metric_function_name, region.id, current_runtime_configuration().remote_retries))
            return pandas.DataFrame(
                [[0.5, 0.25]],
                index=["Salinity (PSU) [sea_water_salinity]{surface}"],
                columns=lead_day_labels(1, 2),
            )

        monkeypatch.setattr(oceanbench.metrics, metric_function_name, metric)
    return calls


def test_evaluate_challenger_writes_the_scores_file_and_prints_the_tables(evaluated_metrics, tmp_path, capsys):
    oceanbench.evaluate_challenger(
        "challenger.py",
        runtime_configuration=RuntimeConfiguration(remote_retries=2),
        region="ibi",
    )

    document = json.loads((tmp_path / "challenger.ibi.scores.json").read_text(encoding="utf-8"))
    output = capsys.readouterr().out
    assert evaluated_metrics == [(metric_function_name, "ibi", 2) for metric_function_name in METRIC_FUNCTION_NAMES]
    assert document["challenger"] == "challenger"
    assert document["region"]["id"] == "ibi"
    assert document["start_dates"] == ["2024-01-03", "2024-01-10"]
    assert document["lead_days_count"] == 2
    assert document["challenger_code"] == CHALLENGER_CODE
    assert list(document["timings"]) == ["open_challenger_dataset", *METRIC_FUNCTION_NAMES, "total"]
    assert len(document["rows"]) == 9 * 2
    assert output.count("Salinity (PSU) [sea_water_salinity]{surface}") == 9
    assert "OceanBench step timings for challenger.ibi.scores.json:" in output


def test_evaluate_challenger_uploads_the_scores_file_to_the_output_bucket(evaluated_metrics, monkeypatch):
    uploads = []

    class FakeS3Client:
        def upload_file(self, file_name, bucket, key, ExtraArgs):
            uploads.append((file_name, bucket, key, ExtraArgs["ContentType"]))

    monkeypatch.setenv("AWS_S3_ENDPOINT", "s3.example.org")
    monkeypatch.setattr(evaluate.boto3, "client", lambda service, endpoint_url: FakeS3Client())

    oceanbench.evaluate_challenger("challenger.py", output_bucket="bucket", output_prefix="dev/reports")

    assert uploads == [
        ("challenger.global.scores.json", "bucket", "dev/reports/challenger.global.scores.json", "application/json")
    ]
