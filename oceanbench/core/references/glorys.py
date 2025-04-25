# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
from typing import List
from xarray import Dataset, open_dataset
import logging


logger = logging.getLogger("copernicusmarine")
logger.setLevel(level=logging.WARNING)


def _glorys_1_4(start_datetime: datetime) -> Dataset:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return open_dataset(
        f"https://minio.dive.edito.eu/project-glonet/public/glorys14_full_2024/{start_datetime_string}.zarr",
        engine="zarr",
    )


def _glorys_datasets(challenger_dataset: Dataset) -> Dataset:
    start_datetime = datetime.fromisoformat(str(challenger_dataset["time"][0].values))
    return _glorys_1_4(start_datetime)


def glorys_datasets(challenger_datasets: List[Dataset]) -> List[Dataset]:
    return list(map(_glorys_datasets, challenger_datasets))
