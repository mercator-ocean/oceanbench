# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import importlib.util
from pathlib import Path

import numpy
import pandas
import pytest

from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.scores import score_rows

_SHARDS_PATH = Path(__file__).resolve().parents[1] / "edito" / "process" / "shards.py"
_METRIC = "rmsd_of_variables_compared_to_glorys_reanalysis"
_LABELS = ["Salinity (PSU) [sea_water_salinity]{surface}", "Temperature (°C) [sea_water_potential_temperature]{50m}"]


@pytest.fixture(scope="module")
def shards():
    specification = importlib.util.spec_from_file_location("edito_shards", _SHARDS_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _start_table(values) -> pandas.DataFrame:
    return pandas.DataFrame(values, index=_LABELS, columns=lead_day_labels(1, 2))


class _FakeShard:
    def __init__(self, start_dates, start_tables) -> None:
        self.start_dates = start_dates
        self.start_tables = start_tables


def test_merged_gridded_rows_are_the_mean_over_the_starts_of_every_shard(shards) -> None:
    first = [_start_table([[1.0, 2.0], [3.0, numpy.nan]]), _start_table([[3.0, 4.0], [5.0, numpy.nan]])]
    second = [_start_table([[5.0, 9.0], [7.0, 1.0]])]
    merged = shards._merged_start_tables([_FakeShard([], {_METRIC: first}), _FakeShard([], {_METRIC: second})])

    rows = shards._gridded_rows(_METRIC, merged[_METRIC])

    expected = score_rows(_METRIC, _start_table([[3.0, 5.0], [5.0, 1.0]]))
    assert [(row["standard_name"], row["depth"], row["lead_day"]) for row in rows] == [
        (row["standard_name"], row["depth"], row["lead_day"]) for row in expected
    ]
    assert [row["value"] for row in rows] == pytest.approx([row["value"] for row in expected])


def test_a_metric_held_by_some_shards_only_is_refused(shards) -> None:
    with pytest.raises(ValueError, match="some shards only"):
        shards._merged_start_tables(
            [_FakeShard([], {_METRIC: [_start_table([[1.0, 2.0], [3.0, 4.0]])]}), _FakeShard([], {})]
        )


@pytest.mark.parametrize(
    "shard_start_dates",
    [
        [["2024-01-10"], ["2024-01-03"]],
        [["2024-01-03"], ["2024-01-17"]],
        [["2024-01-03", "2024-01-10"], ["2024-01-10", "2024-01-17"]],
    ],
)
def test_shards_out_of_order_with_a_gap_or_an_overlap_are_refused(shards, shard_start_dates) -> None:
    with pytest.raises(ValueError, match="start date order"):
        shards._check_start_dates(
            [_FakeShard(dates, {}) for dates in shard_start_dates], ["2024-01-03", "2024-01-10", "2024-01-17"]
        )


def test_the_job_script_and_the_helper_offer_the_same_shard_modes(shards) -> None:
    commands = shards._parser()._subparsers._group_actions[0].choices
    assert set(commands) == {"evaluate-shard", "merge", "eddy-census", "eddy-merge"}
    script = (_SHARDS_PATH.parent / "run-evaluation.sh").read_text()
    for command in commands:
        assert f'"$SHARDS_HELPER" {command}' in script
