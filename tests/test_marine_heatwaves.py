# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray
from numpy.testing import assert_allclose

from oceanbench.core import marine_heatwaves
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.marine_heatwaves import (
    METRIC_LABELS,
    marine_heatwave_diagnostics,
)


FORECAST_LEAD_DAYS = 10
HISTORY_LEAD_DAYS = 7


def _forecast_temperature_dataset(
    temperature_by_pixel: list[float],
    latitudes: list[float],
    lead_day_indices: numpy.ndarray | None = None,
    first_day: str = "2021-06-08",
) -> xarray.Dataset:
    lead_day_indices = numpy.arange(FORECAST_LEAD_DAYS) if lead_day_indices is None else lead_day_indices
    longitudes = list(range(len(temperature_by_pixel) // len(latitudes)))
    coordinates = {
        Dimension.FIRST_DAY_DATETIME.key(): [numpy.datetime64(first_day)],
        Dimension.LEAD_DAY_INDEX.key(): lead_day_indices,
        Dimension.LATITUDE.key(): latitudes,
        Dimension.LONGITUDE.key(): longitudes,
    }
    pixel_temperatures = numpy.array(temperature_by_pixel).reshape(len(latitudes), len(longitudes))
    broadcast_temperatures = numpy.broadcast_to(
        pixel_temperatures,
        (1, lead_day_indices.size, len(latitudes), len(longitudes)),
    )
    return xarray.Dataset(
        data_vars={
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): (
                list(coordinates.keys()),
                numpy.array(broadcast_temperatures, dtype=float),
            )
        },
        coords=coordinates,
    )


def _flat_climatology(
    value: float,
    latitudes: list[float],
    longitude_count: int,
) -> xarray.DataArray:
    day_of_year = numpy.arange(1, 367)
    shape = (day_of_year.size, len(latitudes), longitude_count)
    return xarray.DataArray(
        numpy.full(shape, value),
        dims=["dayofyear", Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={
            "dayofyear": day_of_year,
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): list(range(longitude_count)),
        },
    )


def _score(result, metric_key: str) -> float:
    return float(result.loc[METRIC_LABELS[metric_key]].iloc[0])


def test_detection_scores_follow_the_contingency_table_definitions() -> None:
    latitudes = [0.0]
    warm = 18.0
    cold = 15.0
    challenger = _forecast_temperature_dataset([warm, warm, cold, cold], latitudes)
    reference = _forecast_temperature_dataset([warm, cold, warm, cold], latitudes)
    climatology_mean = _flat_climatology(15.0, latitudes, longitude_count=4)
    percentile_90 = _flat_climatology(16.0, latitudes, longitude_count=4)

    result = marine_heatwave_diagnostics(challenger, reference, climatology_mean, percentile_90)

    true_positive, false_positive, false_negative = 1.0, 1.0, 1.0
    assert_allclose(
        _score(result, "probability_of_detection"),
        true_positive / (true_positive + false_negative),
    )
    assert_allclose(
        _score(result, "false_alarm_ratio"),
        false_positive / (true_positive + false_positive),
    )
    assert_allclose(
        _score(result, "critical_success_index"),
        true_positive / (true_positive + false_positive + false_negative),
    )


def test_detection_scores_are_area_weighted_by_cosine_latitude() -> None:
    latitudes = [0.0, 60.0]
    warm = 18.0
    cold = 15.0
    challenger = _forecast_temperature_dataset([warm, cold, warm, cold], latitudes)
    reference = _forecast_temperature_dataset([warm, cold, cold, warm], latitudes)
    climatology_mean = _flat_climatology(15.0, latitudes, longitude_count=2)
    percentile_90 = _flat_climatology(16.0, latitudes, longitude_count=2)

    result = marine_heatwave_diagnostics(challenger, reference, climatology_mean, percentile_90)

    equator_weight = numpy.cos(numpy.deg2rad(0.0))
    high_latitude_weight = numpy.cos(numpy.deg2rad(60.0))
    true_positive = equator_weight
    false_negative = high_latitude_weight
    assert_allclose(
        _score(result, "probability_of_detection"),
        true_positive / (true_positive + false_negative),
    )


def test_intensity_rmse_uses_anomaly_above_climatological_mean() -> None:
    latitudes = [0.0]
    warm = 18.0
    cold = 15.0
    challenger = _forecast_temperature_dataset([warm, warm, cold, cold], latitudes)
    reference = _forecast_temperature_dataset([warm, cold, warm, cold], latitudes)
    climatology_mean = _flat_climatology(15.0, latitudes, longitude_count=4)
    percentile_90 = _flat_climatology(16.0, latitudes, longitude_count=4)

    result = marine_heatwave_diagnostics(challenger, reference, climatology_mean, percentile_90)

    anomaly = warm - 15.0
    squared_errors_over_union = [0.0, anomaly**2, anomaly**2]
    expected_rmse = numpy.sqrt(numpy.mean(squared_errors_over_union))
    assert_allclose(_score(result, "intensity_rmse"), expected_rmse)


def test_intensity_rmse_depends_on_the_climatology_and_does_not_cancel_out() -> None:
    latitudes = [0.0]
    warm = 18.0
    cold = 15.0
    challenger = _forecast_temperature_dataset([warm, cold], latitudes)
    reference = _forecast_temperature_dataset([cold, warm], latitudes)
    percentile_90 = _flat_climatology(16.0, latitudes, longitude_count=2)

    low_climatology_rmse = _score(
        marine_heatwave_diagnostics(challenger, reference, _flat_climatology(10.0, latitudes, 2), percentile_90),
        "intensity_rmse",
    )
    high_climatology_rmse = _score(
        marine_heatwave_diagnostics(challenger, reference, _flat_climatology(14.0, latitudes, 2), percentile_90),
        "intensity_rmse",
    )

    assert not numpy.isclose(low_climatology_rmse, high_climatology_rmse)


def test_diagnostics_snap_nearly_matching_spatial_coordinates() -> None:
    challenger_latitudes = [0.0, 1.00001, 2.0]
    reference_latitudes = [0.0, 1.0, 2.0]
    warm = 18.0
    challenger = _forecast_temperature_dataset([warm, warm, warm], challenger_latitudes)
    reference = _forecast_temperature_dataset([warm, warm, warm], reference_latitudes)
    climatology_mean = _flat_climatology(15.0, reference_latitudes, longitude_count=1)
    percentile_90 = _flat_climatology(16.0, reference_latitudes, longitude_count=1)

    result = marine_heatwave_diagnostics(challenger, reference, climatology_mean, percentile_90)

    assert_allclose(_score(result, "probability_of_detection"), 1.0)
    assert_allclose(_score(result, "intensity_rmse"), 0.0)


def test_diagnostics_align_when_challenger_has_one_extra_coordinate() -> None:
    matched_latitudes = numpy.linspace(-50.0, 50.0, 1000).tolist()
    challenger_latitudes = matched_latitudes + [51.0]
    warm = 18.0
    challenger = _forecast_temperature_dataset([warm] * len(challenger_latitudes), challenger_latitudes)
    reference = _forecast_temperature_dataset([warm] * len(matched_latitudes), matched_latitudes)
    climatology_mean = _flat_climatology(15.0, matched_latitudes, longitude_count=1)
    percentile_90 = _flat_climatology(16.0, matched_latitudes, longitude_count=1)

    result = marine_heatwave_diagnostics(challenger, reference, climatology_mean, percentile_90)

    assert_allclose(_score(result, "probability_of_detection"), 1.0)
    assert_allclose(_score(result, "intensity_rmse"), 0.0)


def test_events_shorter_than_minimum_duration_are_not_detected() -> None:
    exceedance = numpy.array([True, True, True, True, False, False, False, False, False, False])

    detected = marine_heatwaves._detect_marine_heatwave_events(exceedance, minimum_duration=5, allowed_gap=2)

    assert not detected.any()


def test_events_meeting_minimum_duration_are_detected() -> None:
    exceedance = numpy.array([True, True, True, True, True, False, False, False, False, False])

    detected = marine_heatwaves._detect_marine_heatwave_events(exceedance, minimum_duration=5, allowed_gap=2)

    assert_allclose(detected, exceedance)


def test_internal_gaps_within_the_allowed_gap_are_filled() -> None:
    exceedance = numpy.array([True] * 5 + [False, False] + [True] * 5)

    detected = marine_heatwaves._detect_marine_heatwave_events(exceedance, minimum_duration=5, allowed_gap=2)

    assert detected.all()


def test_gaps_longer_than_the_allowed_gap_keep_events_separated() -> None:
    exceedance = numpy.array([True] * 5 + [False, False, False] + [True] * 5)

    detected = marine_heatwaves._detect_marine_heatwave_events(exceedance, minimum_duration=5, allowed_gap=2)

    assert_allclose(detected, exceedance)


def test_history_extends_detection_without_being_scored() -> None:
    latitudes = [0.0]
    warm = 18.0
    challenger = _forecast_temperature_dataset([warm], latitudes)
    reference = _forecast_temperature_dataset([warm], latitudes)
    history = _forecast_temperature_dataset(
        [warm],
        latitudes,
        lead_day_indices=numpy.arange(-HISTORY_LEAD_DAYS, 0),
    )
    climatology_mean = _flat_climatology(15.0, latitudes, longitude_count=1)
    percentile_90 = _flat_climatology(16.0, latitudes, longitude_count=1)

    result = marine_heatwave_diagnostics(
        challenger,
        reference,
        climatology_mean,
        percentile_90,
        challenger_history_dataset=history,
        reference_history_dataset=history,
    )

    assert result.shape[1] == FORECAST_LEAD_DAYS
    assert_allclose(_score(result, "probability_of_detection"), 1.0)
    assert_allclose(_score(result, "intensity_rmse"), 0.0)
