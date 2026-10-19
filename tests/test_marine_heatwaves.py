# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import xarray

from oceanbench.core.dataset_utils import DepthLevel, Dimension
from oceanbench.core.marine_heatwaves import (
    _leap_year_day_of_year,
    detect_marine_heatwaves,
    marine_heatwave_diagnostics,
    marine_heatwave_scores,
    reference_history_dataset,
)

LEAD_DAYS = 17


def _temperature(values: numpy.ndarray, first_day: str = "2024-06-01") -> xarray.DataArray:
    pixels = values.reshape(-1, LEAD_DAYS)
    return xarray.DataArray(
        pixels.T.reshape(1, LEAD_DAYS, 1, pixels.shape[0]).astype(numpy.float32),
        dims=[
            Dimension.FIRST_DAY_DATETIME.key(),
            Dimension.LEAD_DAY_INDEX.key(),
            Dimension.LATITUDE.key(),
            Dimension.LONGITUDE.key(),
        ],
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): [numpy.datetime64(first_day)],
            Dimension.LEAD_DAY_INDEX.key(): numpy.arange(LEAD_DAYS),
            Dimension.LATITUDE.key(): [0.0],
            Dimension.LONGITUDE.key(): numpy.arange(pixels.shape[0], dtype=float),
        },
    )


def _climatology(
    longitudes: numpy.ndarray,
    seas_by_day: numpy.ndarray | float = 0.0,
    thresh_above_seas: float = 1.0,
) -> xarray.Dataset:
    seas = xarray.DataArray(
        numpy.broadcast_to(seas_by_day, (366,)).astype(float),
        dims=["doy"],
        coords={"doy": numpy.arange(1, 367)},
    ).expand_dims({Dimension.LATITUDE.key(): [0.0], Dimension.LONGITUDE.key(): longitudes})
    return xarray.Dataset({"seas": seas, "thresh": seas + thresh_above_seas})


def test_leap_year_day_of_year_puts_february_29_at_day_60_in_every_year():
    dates = xarray.DataArray(
        pandas.to_datetime(["2023-02-28", "2023-03-01", "2024-02-29", "2024-03-01", "2023-12-31"]).values,
        dims="date",
    )

    assert _leap_year_day_of_year(dates).values.tolist() == [59, 61, 60, 61, 366]


def test_detection_requires_five_days_and_gives_hobday_categories():
    values = numpy.zeros(LEAD_DAYS)
    values[2:8] = [1.5, 2.5, 3.5, 4.5, 1.2, 1.1]
    values[10:13] = 2.0
    temperature = _temperature(values)

    detection = detect_marine_heatwaves(temperature, _climatology(temperature.longitude.values)).squeeze()

    assert numpy.flatnonzero(detection.event.values).tolist() == [2, 3, 4, 5, 6, 7]
    assert detection.category.values[2:8].tolist() == [1, 2, 3, 4, 1, 1]
    assert int(detection.category.values[10:13].sum()) == 0
    numpy.testing.assert_allclose(detection.intensity.values, values, atol=1e-6)


def test_detection_joins_gaps_of_two_days():
    values = numpy.array([2.0] * 5 + [0.0] * 2 + [2.0] * 5 + [0.0] * 5)
    temperature = _temperature(values)

    detection = detect_marine_heatwaves(temperature, _climatology(temperature.longitude.values)).squeeze()

    assert numpy.flatnonzero(detection.event.values).tolist() == list(range(12))


def test_climatology_follows_valid_dates_across_february_29():
    temperature = _temperature(numpy.full(LEAD_DAYS, 100.0), first_day="2024-02-27")
    climatology = _climatology(temperature.longitude.values, seas_by_day=numpy.arange(1, 367))

    intensity = detect_marine_heatwaves(temperature, climatology).intensity.squeeze()

    expected_day_of_year = 58 + numpy.arange(LEAD_DAYS)
    numpy.testing.assert_allclose(intensity.values, 100.0 - expected_day_of_year, atol=1e-6)


def test_missing_pixels_have_no_event_and_no_intensity():
    values = numpy.vstack([numpy.full(LEAD_DAYS, 2.0), numpy.full(LEAD_DAYS, numpy.nan)])
    temperature = _temperature(values)

    detection = detect_marine_heatwaves(temperature, _climatology(temperature.longitude.values)).squeeze()

    assert bool(detection.event.isel(longitude=0).all())
    assert not bool(detection.event.isel(longitude=1).any())
    assert bool(detection.intensity.isel(longitude=1).isnull().all())
    assert int(detection.category.isel(longitude=1).sum()) == 0


def test_dask_chunked_input_gives_the_same_detection():
    random_values = numpy.random.default_rng(0).normal(1.0, 1.0, size=(6, LEAD_DAYS))
    temperature = _temperature(random_values)
    climatology = _climatology(temperature.longitude.values)

    eager = detect_marine_heatwaves(temperature, climatology)
    lazy = detect_marine_heatwaves(temperature.chunk({Dimension.LONGITUDE.key(): 2}), climatology)

    assert lazy.event.chunks is not None
    xarray.testing.assert_identical(eager, lazy.compute())
    reference_temperature = temperature + 0.5
    reference = detect_marine_heatwaves(reference_temperature, climatology)
    xarray.testing.assert_allclose(
        marine_heatwave_scores(temperature, reference_temperature, eager, reference),
        marine_heatwave_scores(
            temperature.chunk({Dimension.LONGITUDE.key(): 2}), reference_temperature, lazy, reference
        ).compute(),
    )


def _field(values, latitudes: list[float]) -> xarray.DataArray:
    grid = numpy.asarray(values, dtype=float)
    lead_days = grid.shape[0]
    return xarray.DataArray(
        grid.reshape(1, lead_days, len(latitudes), -1),
        dims=[
            Dimension.FIRST_DAY_DATETIME.key(),
            Dimension.LEAD_DAY_INDEX.key(),
            Dimension.LATITUDE.key(),
            Dimension.LONGITUDE.key(),
        ],
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): [numpy.datetime64("2024-06-01")],
            Dimension.LEAD_DAY_INDEX.key(): numpy.arange(lead_days),
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): numpy.arange(grid.shape[-1], dtype=float),
        },
    )


def _detection(category, latitudes: list[float]) -> xarray.Dataset:
    category_field = _field(category, latitudes).astype(numpy.int8)
    return xarray.Dataset(
        {"event": category_field > 0, "intensity": category_field.astype(numpy.float32), "category": category_field}
    )


def test_event_and_category_scores_are_area_weighted():
    latitudes = [0.0, 60.0]
    temperature = _field(numpy.zeros((1, 2, 2)), latitudes)
    challenger = _detection([[[2, 3], [0, 4]]], latitudes)
    reference = _detection([[[2, 1], [3, 4]]], latitudes)

    scores = marine_heatwave_scores(temperature, temperature, challenger, reference).squeeze()

    numpy.testing.assert_allclose(
        [scores.probability_of_detection, scores.false_alarm_ratio, scores.critical_success_index],
        [5 / 6, 0.0, 5 / 6],
    )
    numpy.testing.assert_allclose(scores.category_probability_of_detection, [0.75, 0.5, 1.0])
    numpy.testing.assert_allclose(scores.category_false_alarm_ratio, [0.4, 2 / 3, 0.0])
    numpy.testing.assert_allclose(scores.category_critical_success_index, [0.5, 0.25, 1.0])
    numpy.testing.assert_allclose(scores.category_accuracy, 0.5)


def test_temperature_difference_rmse_by_event_agreement_and_bias():
    latitudes = [0.0]
    differences = numpy.array([[[1.0, 2.0, 3.0, 4.0, 9.0]], [[-1.0, -2.0, -3.0, -4.0, 9.0]]])
    challenger_temperature = _field(differences, latitudes).where(lambda field: field != 9.0)
    reference_temperature = _field(numpy.zeros((2, 1, 5)), latitudes)
    challenger = _detection([[[1, 1, 0, 0, 1]], [[1, 1, 0, 0, 1]]], latitudes)
    reference = _detection([[[1, 0, 1, 0, 1]], [[1, 0, 1, 0, 1]]], latitudes)

    scores = marine_heatwave_scores(challenger_temperature, reference_temperature, challenger, reference)

    assert scores.event_agreement.values.tolist() == ["both", "challenger_only", "reference_only", "neither"]
    numpy.testing.assert_allclose(scores.temperature_difference_rmse.squeeze().values, [[1, 2, 3, 4]] * 2)
    numpy.testing.assert_allclose(scores.temperature_bias.squeeze().values, [2.5, -2.5])
    numpy.testing.assert_allclose(scores.probability_of_detection.squeeze().values, [0.5, 0.5])


def test_scores_without_reference_events_are_missing():
    latitudes = [0.0]
    temperature = _field(numpy.zeros((1, 1, 2)), latitudes)
    challenger = _detection([[[1, 0]]], latitudes)
    reference = _detection([[[0, 0]]], latitudes)

    scores = marine_heatwave_scores(temperature, temperature, challenger, reference).squeeze()

    assert bool(scores.probability_of_detection.isnull())
    assert float(scores.false_alarm_ratio) == 1.0
    assert float(scores.critical_success_index) == 0.0


def _window(values, first_days: list[str], lead_days: numpy.ndarray) -> xarray.DataArray:
    return xarray.DataArray(
        numpy.asarray(values, dtype=numpy.float32).reshape(len(first_days), len(lead_days), 1, 1),
        dims=[
            Dimension.FIRST_DAY_DATETIME.key(),
            Dimension.LEAD_DAY_INDEX.key(),
            Dimension.LATITUDE.key(),
            Dimension.LONGITUDE.key(),
        ],
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): numpy.array(first_days, dtype="datetime64[ns]"),
            Dimension.LEAD_DAY_INDEX.key(): lead_days,
            Dimension.LATITUDE.key(): [0.0],
            Dimension.LONGITUDE.key(): [0.0],
        },
    )


def test_reference_history_covers_the_days_before_every_start_including_the_first():
    first_days = ["2024-01-03", "2024-01-10"]
    challenger = _window(numpy.zeros((2, 10)), first_days, numpy.arange(10)).expand_dims(
        {Dimension.DEPTH.key(): [DepthLevel.SURFACE.value, 47.37]}, axis=2
    )
    requests = []

    def day_number_loader(request: xarray.Dataset) -> xarray.Dataset:
        requests.append(request)
        dates = request[Dimension.FIRST_DAY_DATETIME.key()] + request[Dimension.LEAD_DAY_INDEX.key()].astype(
            "timedelta64[D]"
        )
        day_number = (dates - numpy.datetime64("2024-01-01")).dt.days
        return xarray.Dataset({"thetao": day_number.astype(float)})

    history = reference_history_dataset(challenger.to_dataset(name="thetao"), day_number_loader)

    assert requests[0][Dimension.DEPTH.key()].values.tolist() == [DepthLevel.SURFACE.value]
    assert history[Dimension.LEAD_DAY_INDEX.key()].values.tolist() == list(range(-7, 0))
    numpy.testing.assert_array_equal(
        history[Dimension.FIRST_DAY_DATETIME.key()].values, numpy.array(first_days, dtype="datetime64[ns]")
    )
    assert history.thetao.sel(first_day_datetime="2024-01-03").values.tolist() == [-5, -4, -3, -2, -1, 0, 1]
    assert history.thetao.sel(first_day_datetime="2024-01-10").values.tolist() == [2, 3, 4, 5, 6, 7, 8]


def test_history_lets_an_event_in_progress_count_from_lead_day_0_and_is_not_scored():
    first_days = ["2024-06-01"]
    forecast_lead_days = numpy.arange(10)
    forecast = _window([[2.0, 2.0, 2.0] + [0.0] * 7], first_days, forecast_lead_days)
    climatology = _climatology(numpy.array([0.0]))
    history_lead_days = numpy.arange(-7, 0)

    warm_history = _window(numpy.full((1, 7), 2.0), first_days, history_lead_days)
    scores = marine_heatwave_diagnostics(forecast, forecast, warm_history, climatology)

    assert scores[Dimension.LEAD_DAY_INDEX.key()].values.tolist() == forecast_lead_days.tolist()
    assert scores.probability_of_detection.values[:3].tolist() == [1.0, 1.0, 1.0]
    assert bool(scores.probability_of_detection[3:].isnull().all())

    cool_history = _window(numpy.zeros((1, 7)), first_days, history_lead_days)
    scores_without_event = marine_heatwave_diagnostics(forecast, forecast, cool_history, climatology)

    assert bool(scores_without_event.probability_of_detection.isnull().all())
