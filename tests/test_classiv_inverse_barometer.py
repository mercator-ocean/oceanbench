# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pytest
import xarray

from oceanbench.core import classIV_support
from oceanbench.core.classIV_support import (
    CHALLENGER_INVERSE_BAROMETER_VARIABLES,
    REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT,
    _should_use_bracket_vertical_interpolation,
    prepare_class4_model_variable,
)
from oceanbench.core.dataset_source import with_dataset_source
from oceanbench.core.dataset_utils import Dimension, Variable

SEA_SURFACE_HEIGHT_KEY = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
INVERSE_BAROMETER_NAME = "ssh_ib"
CHALLENGER_NAME = "challenger_with_inverse_barometer"

LATITUDES = [-1.0, 0.0, 1.0]
LONGITUDES = [10.0, 11.0]


def _challenger_dataset(with_inverse_barometer: bool, source_name: str | None = CHALLENGER_NAME) -> xarray.Dataset:
    dimensions = [
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    ]
    shape = (1, 2, len(LATITUDES), len(LONGITUDES))
    generator = numpy.random.default_rng(0)
    data_vars = {SEA_SURFACE_HEIGHT_KEY: (dimensions, generator.normal(scale=0.2, size=shape))}
    if with_inverse_barometer:
        data_vars[INVERSE_BAROMETER_NAME] = (dimensions, generator.normal(scale=0.02, size=shape))
    dataset = xarray.Dataset(
        data_vars=data_vars,
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): [numpy.datetime64("2024-01-03")],
            Dimension.LEAD_DAY_INDEX.key(): [0, 1],
            Dimension.LATITUDE.key(): LATITUDES,
            Dimension.LONGITUDE.key(): LONGITUDES,
        },
    )
    if source_name is None:
        return dataset
    return with_dataset_source(dataset, kind="challenger", name=source_name)


@pytest.fixture(autouse=True)
def _without_mean_dynamic_topography(monkeypatch):
    monkeypatch.setattr(classIV_support, "load_mean_dynamic_topography", lambda _resolution: 0.0)


def _registered(monkeypatch) -> None:
    monkeypatch.setitem(CHALLENGER_INVERSE_BAROMETER_VARIABLES, CHALLENGER_NAME, INVERSE_BAROMETER_NAME)


def test_sea_level_anomaly_ignores_the_inverse_barometer_of_an_unregistered_challenger():
    dataset = _challenger_dataset(with_inverse_barometer=True)

    converted = prepare_class4_model_variable(dataset[SEA_SURFACE_HEIGHT_KEY], SEA_SURFACE_HEIGHT_KEY, dataset)

    expected = dataset[SEA_SURFACE_HEIGHT_KEY] - REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT
    numpy.testing.assert_allclose(converted.values, expected.values)


def test_sea_level_anomaly_removes_the_inverse_barometer_of_a_registered_challenger(monkeypatch):
    _registered(monkeypatch)
    dataset = _challenger_dataset(with_inverse_barometer=True)

    converted = prepare_class4_model_variable(dataset[SEA_SURFACE_HEIGHT_KEY], SEA_SURFACE_HEIGHT_KEY, dataset)

    expected = (
        dataset[SEA_SURFACE_HEIGHT_KEY] - dataset[INVERSE_BAROMETER_NAME] - REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT
    )
    numpy.testing.assert_allclose(converted.values, expected.values)


def test_removing_the_inverse_barometer_keeps_the_variable_name_the_interpolator_dispatches_on(monkeypatch):
    _registered(monkeypatch)
    dataset = _challenger_dataset(with_inverse_barometer=True)

    converted = prepare_class4_model_variable(dataset[SEA_SURFACE_HEIGHT_KEY], SEA_SURFACE_HEIGHT_KEY, dataset)

    assert converted.name == SEA_SURFACE_HEIGHT_KEY
    assert _should_use_bracket_vertical_interpolation(str(converted.name))


def test_a_registered_challenger_missing_its_inverse_barometer_is_an_error(monkeypatch):
    _registered(monkeypatch)
    dataset = _challenger_dataset(with_inverse_barometer=False)

    with pytest.raises(ValueError, match=INVERSE_BAROMETER_NAME):
        prepare_class4_model_variable(dataset[SEA_SURFACE_HEIGHT_KEY], SEA_SURFACE_HEIGHT_KEY, dataset)


def test_the_inverse_barometer_is_keyed_on_the_challenger_source_name(monkeypatch):
    _registered(monkeypatch)
    dataset = _challenger_dataset(with_inverse_barometer=True, source_name="another_challenger")

    converted = prepare_class4_model_variable(dataset[SEA_SURFACE_HEIGHT_KEY], SEA_SURFACE_HEIGHT_KEY, dataset)

    expected = dataset[SEA_SURFACE_HEIGHT_KEY] - REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT
    numpy.testing.assert_allclose(converted.values, expected.values)


def test_other_variables_are_not_converted(monkeypatch):
    _registered(monkeypatch)
    dataset = _challenger_dataset(with_inverse_barometer=True)
    temperature_key = Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()

    converted = prepare_class4_model_variable(dataset[SEA_SURFACE_HEIGHT_KEY], temperature_key, dataset)

    numpy.testing.assert_array_equal(converted.values, dataset[SEA_SURFACE_HEIGHT_KEY].values)


def test_no_challenger_declares_an_inverse_barometer_yet():
    assert CHALLENGER_INVERSE_BAROMETER_VARIABLES == {}
