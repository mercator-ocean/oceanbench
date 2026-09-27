# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import gc
import weakref

import numpy
import pytest
import xarray
from numpy.testing import assert_allclose, assert_array_equal
from zarr.errors import GroupNotFoundError

from oceanbench.core import metrics
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.marine_heatwave_history import load_marine_heatwave_analysis_history
from oceanbench.core.marine_heatwaves import METRIC_LABELS
from oceanbench.core.references import glo12, glorys

FIRST_DAY = Dimension.FIRST_DAY_DATETIME.key()
LEAD_DAY = Dimension.LEAD_DAY_INDEX.key()
TEMPERATURE = Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()


def _forecast(first_days=("2024-01-01",), lead_days=2, temperature=18.0):
    coordinates = {
        FIRST_DAY: numpy.array(first_days, dtype="datetime64[ns]"),
        LEAD_DAY: numpy.arange(lead_days),
        Dimension.LATITUDE.key(): [0.0],
        Dimension.LONGITUDE.key(): [0.0],
    }
    return xarray.Dataset(
        {TEMPERATURE: (list(coordinates), numpy.full((len(first_days), lead_days, 1, 1), temperature))},
        coords=coordinates,
    )


def _history(forecast, history_days=7, temperature=18.0):
    history = forecast.isel({LEAD_DAY: slice(0, 0)}).reindex({LEAD_DAY: numpy.arange(-history_days, 0)})
    return xarray.full_like(history, temperature)


def _climatology(value):
    return xarray.DataArray(
        numpy.full((366, 1, 1), value),
        dims=["dayofyear", Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={"dayofyear": numpy.arange(1, 367), Dimension.LATITUDE.key(): [0.0], Dimension.LONGITUDE.key(): [0.0]},
    )


@pytest.mark.parametrize("first_days", [("2024-01-01",), ("2024-01-01", "2024-01-08")])
def test_every_initialization_is_requested_in_one_batch(first_days):
    forecast = _forecast(first_days)
    requests = []

    def loader(request, history_days):
        requests.append(request)
        assert history_days == 7
        return _history(request)

    result = metrics._marine_heatwave_history_dataset(forecast, loader, "global")

    assert requests == [forecast]
    xarray.testing.assert_equal(result, _history(forecast))


@pytest.mark.parametrize("missing_index", [0, 1, 2])
@pytest.mark.parametrize("error_type", [FileNotFoundError, GroupNotFoundError])
def test_partial_availability_retains_other_history_and_forecasts(missing_index, error_type):
    forecast = _forecast(("2024-01-01", "2024-01-08", "2024-01-15"))
    missing_day = forecast[FIRST_DAY].values[missing_index]
    requests = []

    def loader(request, history_days):
        requests.append(request[FIRST_DAY].values.copy())
        if missing_day in request[FIRST_DAY].values:
            raise error_type("unavailable history store")
        return _history(request, history_days)

    result = metrics._marine_heatwave_history_dataset(forecast, loader, "global")

    assert len(requests) == 4
    assert_array_equal(result[FIRST_DAY], forecast[FIRST_DAY])
    assert result[TEMPERATURE].isel({FIRST_DAY: missing_index}).isnull().all()
    available_indices = [index for index in range(3) if index != missing_index]
    assert_allclose(result[TEMPERATURE].isel({FIRST_DAY: available_indices}), 18.0)
    assert_allclose(forecast[TEMPERATURE], 18.0)


@pytest.mark.parametrize("first_days", [("2024-01-01",), ("2024-01-01", "2024-01-08")])
def test_no_available_history_returns_none(first_days):
    forecast = _forecast(first_days)

    def loader(request, history_days):
        raise FileNotFoundError("unavailable history store")

    assert metrics._marine_heatwave_history_dataset(forecast, loader, "global") is None


@pytest.mark.parametrize("error_type", [PermissionError, ConnectionError, TimeoutError, ValueError, KeyError])
def test_unrelated_loader_errors_propagate(error_type):
    error = error_type("cannot load history")

    def loader(request, history_days):
        raise error

    with pytest.raises(error_type) as caught:
        metrics._marine_heatwave_history_dataset(_forecast(), loader, "global")
    assert caught.value is error


@pytest.mark.parametrize("outer_error_type", [FileNotFoundError, GroupNotFoundError])
@pytest.mark.parametrize("cause_type", [PermissionError, ConnectionError, ValueError])
def test_absence_shaped_errors_do_not_hide_authentication_transport_or_corrupt_metadata(outer_error_type, cause_type):
    error = outer_error_type("cannot open store")
    error.__cause__ = cause_type("underlying failure")

    def loader(request, history_days):
        raise error

    with pytest.raises(outer_error_type) as caught:
        metrics._marine_heatwave_history_dataset(_forecast(), loader, "global")
    assert caught.value is error


def test_operational_errors_during_individual_retries_propagate():
    forecast = _forecast(("2024-01-01", "2024-01-08"))

    def loader(request, history_days):
        if request.sizes[FIRST_DAY] == 2:
            raise FileNotFoundError("one unavailable history store")
        raise ConnectionError("history server disconnected")

    with pytest.raises(ConnectionError, match="disconnected"):
        metrics._marine_heatwave_history_dataset(forecast, loader, "global")


def test_public_zarr_missing_metadata_exception_chain_falls_back():
    error = FileNotFoundError("absent Zarr store")
    group_error = GroupNotFoundError("")
    metadata_error = KeyError(".zmetadata")
    error.__cause__ = group_error
    group_error.__cause__ = metadata_error
    metadata_error.__cause__ = FileNotFoundError("absent .zmetadata")

    def loader(request, history_days):
        raise error

    assert metrics._marine_heatwave_history_dataset(_forecast(), loader, "global") is None


@pytest.mark.parametrize("lead_days", [1, 2])
def test_analysis_history_requests_seven_days_for_a_short_forecast(lead_days):
    forecast = _forecast(("2024-01-01", "2024-01-08"), lead_days=lead_days)
    requests = []

    def analysis_loader(request):
        requests.append(request)
        return xarray.full_like(request, 18.0)

    result = load_marine_heatwave_analysis_history(forecast, analysis_loader, history_days=7)

    assert len(requests) == 1
    assert_array_equal(requests[0][FIRST_DAY], forecast[FIRST_DAY] - numpy.timedelta64(7, "D"))
    assert_array_equal(requests[0][LEAD_DAY], numpy.arange(7))
    assert_array_equal(result[FIRST_DAY], forecast[FIRST_DAY])
    assert_array_equal(result[LEAD_DAY], numpy.arange(-7, 0))
    assert_allclose(result[TEMPERATURE], 18.0)


def test_analysis_history_shape_errors_propagate():
    def analysis_loader(request):
        return request.isel({LEAD_DAY: slice(0, 2)})

    with pytest.raises(ValueError):
        load_marine_heatwave_analysis_history(_forecast(), analysis_loader, history_days=7)


@pytest.mark.parametrize("available_product", ["challenger", "reference", "both", "neither"])
@pytest.mark.parametrize("lead_days", [1, 2])
def test_independent_product_histories_are_preserved_and_only_forecast_days_are_scored(
    monkeypatch, available_product, lead_days
):
    forecast = _forecast(lead_days=lead_days)
    history = _history(forecast)
    monkeypatch.setattr(
        metrics,
        "marine_heatwave_climatology_mean_and_percentile_90",
        lambda dataset: (_climatology(15), _climatology(16)),
    )
    result = metrics._marine_heatwave_diagnostics_against_reference(
        forecast,
        forecast,
        history if available_product in ("challenger", "both") else None,
        history if available_product in ("reference", "both") else None,
        "global",
    )

    assert result.shape[1] == lead_days
    if available_product == "both":
        assert_allclose(result.loc[METRIC_LABELS["probability_of_detection"]], 1.0)
        assert_allclose(result.loc[METRIC_LABELS["intensity_rmse"]], 0.0)
    elif available_product == "challenger":
        assert_allclose(result.loc[METRIC_LABELS["false_alarm_ratio"]], 1.0)
    elif available_product == "reference":
        assert_allclose(result.loc[METRIC_LABELS["probability_of_detection"]], 0.0)
    else:
        assert result.isna().all().all()


def test_forecast_can_still_detect_an_event_when_both_histories_are_unavailable(monkeypatch):
    forecast = _forecast(lead_days=5)
    monkeypatch.setattr(
        metrics,
        "marine_heatwave_climatology_mean_and_percentile_90",
        lambda dataset: (_climatology(15), _climatology(16)),
    )
    result = metrics._marine_heatwave_diagnostics_against_reference(forecast, forecast, None, None, "global")

    assert result.shape[1] == 5
    assert_allclose(result.loc[METRIC_LABELS["probability_of_detection"]], 1.0)
    assert_allclose(result.loc[METRIC_LABELS["intensity_rmse"]], 0.0)


@pytest.mark.parametrize(
    "reference_module,cache_name,loader_name,remote_loader_name",
    [
        (glo12, "_GLO12_ANALYSIS_DATASET_CACHE", "glo12_analysis_dataset", "_glo12_analysis_dataset_1_degree"),
        (
            glorys,
            "_GLORYS_REANALYSIS_DATASET_CACHE",
            "glorys_reanalysis_dataset",
            "_glorys_reanalysis_dataset_1_degree",
        ),
    ],
)
@pytest.mark.parametrize("expired_request", [False, True])
def test_reference_cache_checks_request_identity_even_when_python_ids_are_reused(
    monkeypatch, reference_module, cache_name, loader_name, remote_loader_name, expired_request
):
    forecast = _forecast()
    old_request = _forecast()
    old_request_reference = weakref.ref(old_request)
    if expired_request:
        del old_request
        gc.collect()
        assert old_request_reference() is None
    stale_reference = xarray.full_like(forecast, 2.0)
    cache = {id(forecast): (old_request_reference, stale_reference)}
    monkeypatch.setattr(reference_module, cache_name, cache)
    monkeypatch.setattr(reference_module, "get_dataset_resolution", lambda request: "one_degree")
    monkeypatch.setattr(reference_module, "with_remote_http_retries", lambda operation, callback: callback())
    requests = []

    def remote_loader(request):
        requests.append(weakref.ref(request))
        return xarray.full_like(request, 18.0)

    monkeypatch.setattr(reference_module, remote_loader_name, remote_loader)
    loader = getattr(reference_module, loader_name)

    reference = loader(forecast)
    assert_allclose(reference[TEMPERATURE], 18.0)
    assert loader(forecast) is reference
    assert len(requests) == 1
    assert cache[id(forecast)][0]() is forecast
    request_identity = id(forecast)
    del forecast
    gc.collect()
    assert request_identity not in cache


@pytest.mark.parametrize("available_product", ["challenger", "reference", "both", "neither"])
def test_public_glorys_wrapper_keeps_first_initialization_and_independent_histories(monkeypatch, available_product):
    forecast = _forecast()
    monkeypatch.setattr(metrics, "marine_heatwave_climatology_is_available", lambda dataset: True)
    monkeypatch.setattr(metrics, "glorys_reanalysis_dataset", lambda dataset: forecast)
    monkeypatch.setattr(
        metrics,
        "marine_heatwave_climatology_mean_and_percentile_90",
        lambda dataset: (_climatology(15), _climatology(16)),
    )
    requests = []

    def loader(product):
        def load_history(request, history_days):
            assert_array_equal(request[FIRST_DAY], forecast[FIRST_DAY])
            requests.append(product)
            if available_product not in (product, "both"):
                raise FileNotFoundError("unavailable history store")
            return _history(request, history_days)

        return load_history

    monkeypatch.setattr(metrics, "glo12_analysis_history_dataset", loader("challenger"))
    monkeypatch.setattr(metrics, "glorys_reanalysis_history_dataset", loader("reference"))

    result = metrics.marine_heatwave_diagnostics_compared_to_glorys_reanalysis(forecast)

    assert requests == ["challenger", "reference"]
    assert result.shape[1] == 2
    if available_product == "challenger":
        assert_allclose(result.loc[METRIC_LABELS["false_alarm_ratio"]], 1.0)
    elif available_product == "reference":
        assert_allclose(result.loc[METRIC_LABELS["probability_of_detection"]], 0.0)
    elif available_product == "both":
        assert_allclose(result.loc[METRIC_LABELS["probability_of_detection"]], 1.0)
    else:
        assert result.isna().all().all()


def test_real_missing_zarr_history_store_falls_back(tmp_path):
    def loader(request, history_days):
        return xarray.open_dataset(tmp_path / "missing.zarr", engine="zarr")

    assert metrics._marine_heatwave_history_dataset(_forecast(), loader, "global") is None


def test_real_corrupt_zarr_history_metadata_propagates(tmp_path):
    store = tmp_path / "corrupt.zarr"
    store.mkdir()
    (store / ".zgroup").write_text("invalid JSON", encoding="utf8")

    def loader(request, history_days):
        return xarray.open_dataset(store, engine="zarr")

    with pytest.raises(ValueError):
        metrics._marine_heatwave_history_dataset(_forecast(), loader, "global")
