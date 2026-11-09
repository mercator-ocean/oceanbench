# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas

from oceanbench.core.classIV import CLASS4_MATCHUP_COLUMNS, class4_rmsd_and_matchups, rmsd_class4_validation
from oceanbench.core.classIV_support import _CLASS4_OBSERVATIONS_CACHE
from oceanbench.core.lead_day_utils import lead_day_labels
from tests import synthetic_class4

_TEMPERATURE = "Temperature (°C) [sea_water_potential_temperature]"
_SALINITY = "Salinity (PSU) [sea_water_salinity]"
# Class IV table of the synthetic data computed by the code of main 210f1e2.
_EXPECTED_CLASS4_TABLE = pandas.DataFrame(
    [
        [numpy.nan, numpy.nan, 1.6351224696048696, numpy.nan, numpy.nan],
        [1.3761407723557493, 1.8265336631826121, numpy.nan, 2.0, 0.0],
        [1.0331997915188074, 0.8331441979294442, 0.9864481333554291, 33.0, 3.0],
        [1.154034653719385, 1.068165555464438, 0.9937730285717536, 58.0, 2.0],
        [1.2648846142294516, 1.2066023682196694, 1.0104602872852257, 77.0, 4.0],
        [0.18820749300503686, 0.172703125371561, 0.00434269130126097, 2.0, 0.0],
        [0.08840498180133982, 0.12642939915899065, 0.10272910136555183, 33.0, 3.0],
        [0.11302024382689588, 0.1197397717004122, 0.10710207160250156, 58.0, 2.0],
        [0.1187089333700049, 0.12820777852550716, 0.11783319096054798, 77.0, 4.0],
    ],
    index=[
        f"{_TEMPERATURE}{{surface}}",
        f"{_TEMPERATURE}{{0-5m}}",
        f"{_TEMPERATURE}{{5-100m}}",
        f"{_TEMPERATURE}{{100-300m}}",
        f"{_TEMPERATURE}{{300-600m}}",
        f"{_SALINITY}{{0-5m}}",
        f"{_SALINITY}{{5-100m}}",
        f"{_SALINITY}{{100-300m}}",
        f"{_SALINITY}{{300-600m}}",
    ],
    columns=["Lead day 1", "Lead day 2", "Lead day 3", "Observations", "Missing"],
)


def _class4_arguments() -> tuple:
    return (
        synthetic_class4.challenger_dataset(),
        synthetic_class4.observations_dataset(),
        synthetic_class4.ocean_mask(),
        synthetic_class4.VARIABLES,
    )


def _class4_table() -> pandas.DataFrame:
    return rmsd_class4_validation(*_class4_arguments())


def _matchups() -> pandas.DataFrame:
    return class4_rmsd_and_matchups(*_class4_arguments(), keep_matchups=True)[1]


def test_class4_scores_are_the_scores_of_main() -> None:
    pandas.testing.assert_frame_equal(_class4_table(), _EXPECTED_CLASS4_TABLE, check_exact=False, rtol=1e-12)


def test_keeping_the_matchups_does_not_change_the_class4_scores() -> None:
    table, _ = class4_rmsd_and_matchups(*_class4_arguments(), keep_matchups=True)

    pandas.testing.assert_frame_equal(table, _class4_table(), check_exact=True)


def test_matchups_hold_the_filtered_population_with_its_missing_challenger_values() -> None:
    matchups = _matchups()

    assert list(matchups.columns) == CLASS4_MATCHUP_COLUMNS
    assert matchups["model_value"].isna().any()
    assert 0 < len(matchups) < 2 * 600
    table = _class4_table()
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
