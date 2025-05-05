# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
import numpy
from xarray import Dataset, open_mfdataset
import logging


logger = logging.getLogger("copernicusmarine")
logger.setLevel(level=logging.WARNING)


def _glorys_1_4_path(start_datetime: numpy.datetime64) -> str:
    start_datetime_string = datetime.fromisoformat(str(start_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-glonet/public/glorys14_refull_2024/{start_datetime_string}.zarr"


def glorys_dataset(challenger_dataset: Dataset) -> Dataset:

    start_datetimes = challenger_dataset["start_datetime"].values
    return open_mfdataset(
        list(map(_glorys_1_4_path, start_datetimes)),
        engine="zarr",
        preprocess=lambda dataset: dataset.assign(time=range(10)),
        combine="nested",
        concat_dim="start_datetime",
        parallel=True,
    ).assign(start_datetime=start_datetimes)
