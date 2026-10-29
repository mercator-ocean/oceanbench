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
        "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/public/ifs-nowcast-corrected/20240102.zarr"
    )


def test_ifs_nowcasts_concatenates_four_steps_per_week(monkeypatch) -> None:
    datetimes = [datetime(2024, 1, 2), datetime(2024, 1, 9)]
    monkeypatch.setattr(input_datasets, "_ifs_nowcast_datetimes", lambda: datetimes)
    opened_paths = []

    def fake_open(dataset_path):
        opened_paths.append(dataset_path)
        date_string = dataset_path.rsplit("/", 1)[-1][:8]
        start_datetime = datetime.strptime(date_string, "%Y%m%d")
        times = numpy.array([numpy.datetime64(start_datetime + timedelta(hours=6 * step)) for step in range(4)])
        data_vars = {}
        for variable in input_datasets._IFS_NOWCAST_ANALYSIS_VARIABLES:
            data_vars[variable] = (("time", "lat", "lon"), numpy.ones((4, 1, 1)))
        for variable in input_datasets._IFS_NOWCAST_ACCUMULATED_VARIABLES:
            data_vars[variable] = (("time", "lat", "lon"), numpy.ones((4, 1, 1)))
        return xarray.Dataset(data_vars, coords={"time": times, "lat": [0.0], "lon": [0.0]})

    monkeypatch.setattr(input_datasets, "_open_ifs_nowcast_dataset", fake_open)

    dataset = input_datasets.ifs_nowcasts()

    assert dataset.sizes == {"time": 8, "lat": 1, "lon": 1}
    assert set(dataset.data_vars) == set(
        input_datasets._IFS_NOWCAST_ANALYSIS_VARIABLES + input_datasets._IFS_NOWCAST_ACCUMULATED_VARIABLES
    )
    assert opened_paths == [
        f"{input_datasets._IFS_NOWCAST_DATASET_ROOT}/20240102.zarr",
        f"{input_datasets._IFS_NOWCAST_DATASET_ROOT}/20240109.zarr",
    ]
    assert not dataset["sosudosw"].isnull().any().item()
    assert not dataset["sotemair"].isnull().any().item()
