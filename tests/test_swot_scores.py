# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
import xarray

from oceanbench.core import classIV_support, metrics
from oceanbench.core.classIV_support import format_class4_results
from oceanbench.core.references.observations import ObservationDataUnavailableError
from oceanbench.core.scores import score_rows

LEAD_DAYS_COUNT = 10
SWOT_VARIABLE = "high resolution sea level anomaly (swot)"


def _class4_scores(variables_and_depth_bins: list[tuple[str, str]]) -> pandas.DataFrame:
    results = pandas.DataFrame(
        [
            (variable, depth_bin, lead_day, 0.05 + lead_day / 100, 1000, 0)
            for variable, depth_bin in variables_and_depth_bins
            for lead_day in range(LEAD_DAYS_COUNT)
        ],
        columns=["variable", "depth_bin", "lead_day", "rmsd", "count", "missing"],
    )
    return format_class4_results(results, LEAD_DAYS_COUNT)


def _nadir_and_in_situ_scores() -> pandas.DataFrame:
    return _class4_scores(
        [("sea_water_potential_temperature", "surface"), ("sea_surface_height_above_geoid", "surface")]
    )


def _swot_class4_scores() -> pandas.DataFrame:
    return _class4_scores([("sea_surface_height_above_geoid", "surface")])


def _use_fake_class4_inputs(monkeypatch, swot_observations) -> list[xarray.Dataset]:
    nadir_observations_dataset = xarray.Dataset()
    swot_observations_dataset = xarray.Dataset()
    reference_datasets = []

    def fake_rmsd_class4_validation(challenger_dataset, reference_dataset, ocean_mask, variables):
        reference_datasets.append(reference_dataset)
        return _nadir_and_in_situ_scores() if reference_dataset is nadir_observations_dataset else _swot_class4_scores()

    monkeypatch.setattr(metrics, "observations", lambda _challenger_dataset: nadir_observations_dataset)
    monkeypatch.setattr(metrics, "swot_observations", swot_observations(swot_observations_dataset))
    monkeypatch.setattr(metrics, "ocean_mask", lambda: xarray.DataArray())
    monkeypatch.setattr(metrics, "rmsd_class4_validation", fake_rmsd_class4_validation)
    return reference_datasets


def _unavailable_swot_observations(_swot_observations_dataset):
    def raise_unavailable(_challenger_dataset):
        raise ObservationDataUnavailableError("No converted SWOT observation day covers its forecast period.")

    return raise_unavailable


def test_swot_score_label_is_parsed_as_its_own_variable():
    swot_scores = _swot_class4_scores().rename(index=metrics._swot_score_label)

    swot_rows = score_rows("rmsd_of_variables_compared_to_observations", swot_scores)

    assert {(row["variable"], row["standard_name"], row["depth"], row["unit"]) for row in swot_rows} == {
        (SWOT_VARIABLE, "sea_surface_height_above_geoid", "surface", "m")
    }


def test_observation_scores_end_with_the_swot_row(monkeypatch):
    _use_fake_class4_inputs(monkeypatch, lambda dataset: lambda _challenger_dataset: dataset)

    scores = metrics.rmsd_of_variables_compared_to_observations(xarray.Dataset())

    assert list(scores.index) == [
        "Temperature (°C) [sea_water_potential_temperature]{surface}",
        "Sea level anomaly (m) [sea_surface_height_above_geoid]{surface}",
        "High resolution sea level anomaly (SWOT) (m) [sea_surface_height_above_geoid]{surface}",
    ]


def test_observation_scores_without_swot_observations_keep_the_other_rows(monkeypatch):
    reference_datasets = _use_fake_class4_inputs(monkeypatch, _unavailable_swot_observations)

    scores = metrics.rmsd_of_variables_compared_to_observations(xarray.Dataset())

    assert len(reference_datasets) == 1
    score_variables = {row["variable"] for row in score_rows("rmsd_of_variables_compared_to_observations", scores)}
    assert SWOT_VARIABLE not in score_variables
    assert len(scores) == 2


def test_class4_observations_cache_is_released_after_each_score_table(monkeypatch):
    _use_fake_class4_inputs(monkeypatch, lambda dataset: lambda _challenger_dataset: dataset)
    classIV_support._CLASS4_OBSERVATIONS_CACHE[(0, LEAD_DAYS_COUNT)] = (pandas.DataFrame(), None, "observations")

    metrics.rmsd_of_variables_compared_to_observations(xarray.Dataset())

    assert classIV_support._CLASS4_OBSERVATIONS_CACHE == {}
