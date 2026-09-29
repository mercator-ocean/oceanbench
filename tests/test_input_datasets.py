# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime, timedelta

import xarray
import numpy

from oceanbench.core import input_datasets


def test_ifs_nowcast_datetimes_cover_2024_weekly() -> None:
    datetimes = input_datasets._ifs_nowcast_datetimes()

    assert len(datetimes) == 52
    assert datetimes[0] == datetime(2024, 1, 2)
    assert datetimes[-1] == datetime(2024, 12, 24)
    assert all(dt.weekday() == 1 for dt in datetimes)


def test_ifs_nowcast_dataset_path_uses_published_cloudferro_public_location() -> None:
    assert input_datasets._ifs_nowcast_dataset_path(datetime(2024, 1, 2)) == (
        f"{input_datasets._IFS_NOWCAST_DATASET_ROOT}/20240102.zarr"
    )


def test_ifs_nowcasts_concatenates_root_stores_and_preserves_terminal_nan(monkeypatch) -> None:
    datetimes = [datetime(2024, 1, 2), datetime(2024, 1, 9)]
    monkeypatch.setattr(input_datasets, "_ifs_nowcast_datetimes", lambda: datetimes)
    opened_paths = []

    def fake_open(dataset_path):
        opened_paths.append(dataset_path)
        date_string = dataset_path.rsplit("/", 1)[-1][:8]
        start_datetime = datetime.strptime(date_string, "%Y%m%d")
        times = numpy.array([numpy.datetime64(start_datetime + timedelta(hours=6 * step)) for step in range(5)])
        data_vars = {}
        for variable in input_datasets._IFS_NOWCAST_ANALYSIS_VARIABLES:
            data_vars[variable] = (("time", "lat", "lon"), numpy.ones((5, 1, 1)))
        for variable in input_datasets._IFS_NOWCAST_ACCUMULATED_VARIABLES:
            values = numpy.ones((5, 1, 1))
            values[-1] = numpy.nan
            data_vars[variable] = (("time", "lat", "lon"), values)
        return xarray.Dataset(data_vars, coords={"time": times, "lat": [0.0], "lon": [0.0]})

    monkeypatch.setattr(input_datasets, "_open_ifs_nowcast_dataset", fake_open)

    dataset = input_datasets.ifs_nowcasts()

    assert dataset.sizes == {"time": 10, "lat": 1, "lon": 1}
    assert set(dataset.data_vars) == set(
        input_datasets._IFS_NOWCAST_ANALYSIS_VARIABLES + input_datasets._IFS_NOWCAST_ACCUMULATED_VARIABLES
    )
    assert opened_paths == [
        f"{input_datasets._IFS_NOWCAST_DATASET_ROOT}/20240102.zarr",
        f"{input_datasets._IFS_NOWCAST_DATASET_ROOT}/20240109.zarr",
    ]
    assert dataset["sosudosw"].isel(time=4).isnull().all().item()
    assert dataset["sosudosw"].isel(time=9).isnull().all().item()
    assert not dataset["sotemair"].isel(time=4).isnull().any().item()


def test_ifs_nowcast_repairs_missing_first_longitude() -> None:
    dataset = xarray.Dataset(coords={"lon": [numpy.nan, 359.93]})
    repaired = input_datasets._repair_ifs_nowcast_longitude(dataset)
    numpy.testing.assert_allclose(repaired["lon"], [0.0, 359.93])
