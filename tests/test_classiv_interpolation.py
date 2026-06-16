# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import dask.array
import numpy
import pandas
import xarray

from oceanbench.core.classIV_support import interpolate_class4_model_to_observations
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.runtime_configuration import RuntimeConfiguration
import oceanbench.core.runtime_configuration as runtime_configuration


def _model_data() -> xarray.DataArray:
    first_days = numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]")
    lead_days = numpy.array([0, 1, 2])
    depths = numpy.array([0.0])
    latitudes = numpy.array([0.0, 1.0])
    longitudes = numpy.array([10.0, 11.0])
    values = numpy.empty((len(first_days), len(lead_days), len(depths), len(latitudes), len(longitudes)))

    for first_day_index in range(len(first_days)):
        for lead_day_index, lead_day in enumerate(lead_days):
            for latitude_index, latitude in enumerate(latitudes):
                for longitude_index, longitude in enumerate(longitudes):
                    values[first_day_index, lead_day_index, 0, latitude_index, longitude_index] = (
                        100 * first_day_index + 10 * lead_day + latitude + 2 * (longitude - 10)
                    )

    return xarray.DataArray(
        dask.array.from_array(values, chunks=(1, 1, 1, len(latitudes), len(longitudes))),
        dims=[
            Dimension.FIRST_DAY_DATETIME.key(),
            Dimension.LEAD_DAY_INDEX.key(),
            Dimension.DEPTH.key(),
            Dimension.LATITUDE.key(),
            Dimension.LONGITUDE.key(),
        ],
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): first_days,
            Dimension.LEAD_DAY_INDEX.key(): lead_days,
            Dimension.DEPTH.key(): depths,
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
        name=Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
    )


def _observations_dataframe() -> pandas.DataFrame:
    first_days = numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]")
    return pandas.DataFrame(
        {
            Dimension.TIME.key(): pandas.to_datetime(
                ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-10", "2024-01-12"]
            ),
            Dimension.LATITUDE.key(): [0.0, 0.5, 1.0, 0.25, 0.75],
            Dimension.LONGITUDE.key(): [10.0, 10.5, 11.0, 10.25, 10.75],
            "first_day": [first_days[0], first_days[0], first_days[0], first_days[1], first_days[1]],
            Dimension.DEPTH.key(): [0.0] * 5,
            "lead_day": [0, 1, 2, 0, 2],
            "observation_value": [0.0] * 5,
        }
    )


def _record_first_day_block_compute_calls(monkeypatch) -> list[tuple[int, ...]]:
    original_compute = xarray.DataArray.compute
    first_day_block_compute_calls = []

    def record_compute(data_array: xarray.DataArray, *args, **kwargs):
        if (
            Dimension.LEAD_DAY_INDEX.key() in data_array.dims
            and Dimension.FIRST_DAY_DATETIME.key() not in data_array.dims
        ):
            first_day_block_compute_calls.append(tuple(data_array[Dimension.LEAD_DAY_INDEX.key()].values.tolist()))
        return original_compute(data_array, *args, **kwargs)

    monkeypatch.setattr(xarray.DataArray, "compute", record_compute)
    return first_day_block_compute_calls


def test_class4_model_interpolation_uses_memory_safe_materialization_by_default(monkeypatch) -> None:
    first_day_block_compute_calls = _record_first_day_block_compute_calls(monkeypatch)

    model_values = interpolate_class4_model_to_observations(_model_data(), _observations_dataframe())

    numpy.testing.assert_allclose(model_values, [0.0, 11.5, 23.0, 100.75, 122.25])
    assert first_day_block_compute_calls == []


def test_class4_fast_interpolation_materializes_each_first_day_block_once(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_configuration,
        "_runtime_configuration",
        RuntimeConfiguration(class4_fast_interpolation=True),
    )
    first_day_block_compute_calls = _record_first_day_block_compute_calls(monkeypatch)

    model_values = interpolate_class4_model_to_observations(_model_data(), _observations_dataframe())

    numpy.testing.assert_allclose(model_values, [0.0, 11.5, 23.0, 100.75, 122.25])
    assert first_day_block_compute_calls == [(0, 1, 2), (0, 1, 2)]
