# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas

from oceanbench.core.classIV import CLASS4_MATCHUP_COLUMNS, class4_matchups, class4_rmsd_table, rmsd_class4_validation
from oceanbench.core.classIV_support import _CLASS4_OBSERVATIONS_CACHE
from oceanbench.core.lead_day_utils import lead_day_labels
from tests import synthetic_class4


def _matchups() -> pandas.DataFrame:
    return class4_matchups(
        synthetic_class4.challenger_dataset(),
        synthetic_class4.observations_dataset(),
        synthetic_class4.ocean_mask(),
        synthetic_class4.VARIABLES,
    )


def test_class4_scores_are_computed_from_the_matchups() -> None:
    matchups = _matchups()

    pandas.testing.assert_frame_equal(
        class4_rmsd_table(matchups, synthetic_class4.LEAD_DAYS_COUNT),
        rmsd_class4_validation(
            synthetic_class4.challenger_dataset(),
            synthetic_class4.observations_dataset(),
            synthetic_class4.ocean_mask(),
            synthetic_class4.VARIABLES,
        ),
    )


def test_matchups_hold_the_filtered_population_with_its_missing_challenger_values() -> None:
    matchups = _matchups()

    assert list(matchups.columns) == CLASS4_MATCHUP_COLUMNS
    assert matchups["model_value"].isna().any()
    assert 0 < len(matchups) < 2 * 600
    table = class4_rmsd_table(matchups, synthetic_class4.LEAD_DAYS_COUNT)
    for (variable, depth_bin), group in matchups.groupby(["variable", "depth_bin"]):
        row = next(label for label in table.index if f"[{variable}]{{{depth_bin}}}" in label)
        for lead_day, lead_group in group.groupby("lead_day"):
            scored = lead_group.dropna(subset=["model_value"])
            pooled = numpy.sqrt(((scored["model_value"] - scored["observation_value"]) ** 2).mean())
            numpy.testing.assert_allclose(
                table.at[row, lead_day_labels(1, synthetic_class4.LEAD_DAYS_COUNT)[lead_day]], pooled, rtol=1e-12
            )
        # Counts are reported at the first lead day of the whole table.
        first_lead_day = group[group["lead_day"] == matchups["lead_day"].min()]
        if len(first_lead_day):
            assert table.at[row, "Observations"] == len(first_lead_day)
            assert table.at[row, "Missing"] == first_lead_day["model_value"].isna().sum()


def test_each_class4_run_starts_with_an_empty_observation_cache() -> None:
    _CLASS4_OBSERVATIONS_CACHE[(0, 0)] = (pandas.DataFrame(), numpy.array([]), "observation")

    _matchups()

    assert (0, 0) not in _CLASS4_OBSERVATIONS_CACHE
