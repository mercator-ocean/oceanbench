# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
import numpy
from xarray import Dataset, open_mfdataset
import logging


logger = logging.getLogger("copernicusmarine")
logger.setLevel(level=logging.WARNING)

from oceanbench.core.dataset_utils import Dimension


def _glo12_1_4_path(first_day_datetime: numpy.datetime64) -> str:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/glo14/{first_day}.zarr"


def glo12_analysis_dataset(challenger_dataset: Dataset) -> Dataset:

    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    return open_mfdataset(
        list(map(_glo12_1_4_path, first_day_datetimes)),
        engine="zarr",
        preprocess=lambda dataset: dataset.rename({"time": Dimension.LEAD_DAY_INDEX.key()}).assign(
            {Dimension.LEAD_DAY_INDEX.key(): range(10)}
        ),
        combine="nested",
        concat_dim=Dimension.FIRST_DAY_DATETIME.key(),
        parallel=True,
    ).assign({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes})
