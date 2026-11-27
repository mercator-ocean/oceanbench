# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import sys

import pandas
import pytest

import oceanbench
import oceanbench.core.metrics as core_metrics
import oceanbench.core.runtime_configuration as runtime_configuration
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.runtime_configuration import current_runtime_configuration
from oceanbench.core.scores import GRIDDED_METRIC_FUNCTION_NAMES, METRIC_FUNCTION_NAMES

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
        if metric_function_name in GRIDDED_METRIC_FUNCTION_NAMES:
            # With a viewer, the scoring run keeps the table of each of the two start dates.
            def metric_with_start_tables(challenger_dataset, region, keep_start_tables, metric=metric):
                table = metric(challenger_dataset, region)
                return table, [table * start_number for start_number in (1, 2)]

            monkeypatch.setattr(core_metrics, metric_function_name, metric_with_start_tables)

    def _gridded_rmsd_metrics_together(metric_function_names, challenger_dataset, region, keep_start_tables=False):
        # The stubs above, as evaluate computes some gridded metrics together.
        return {
            name: (
                getattr(core_metrics, name)(challenger_dataset, region, keep_start_tables=True)
                if keep_start_tables
                else getattr(oceanbench.metrics, name)(challenger_dataset, region)
            )
            for name in metric_function_names
        }

    monkeypatch.setattr(core_metrics, "_gridded_rmsd_metrics_together", _gridded_rmsd_metrics_together)
    return calls
