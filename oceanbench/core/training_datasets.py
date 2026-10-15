# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
import xarray

from oceanbench.core.references.glorys import open_glorys_twelfth_degree_reanalysis
from oceanbench.core.remote_http import open_remote_multizarr

GLORYS_TRAINING_FIRST_DAY = "1993-01-01"
GLORYS_TRAINING_LAST_DAY = "2019-12-31"
GLORYS_ONE_DEGREE_BUCKET_URL = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/public/glorys_1degree_1993_2019"


def _glorys_one_degree_monthly_paths() -> list[str]:
    monthly_dates = pandas.date_range(GLORYS_TRAINING_FIRST_DAY, GLORYS_TRAINING_LAST_DAY, freq="MS")
    return [f"{GLORYS_ONE_DEGREE_BUCKET_URL}/{monthly_date.strftime('%Y%m')}.zarr" for monthly_date in monthly_dates]


def glorys_reanalysis() -> xarray.Dataset:
    return open_glorys_twelfth_degree_reanalysis(GLORYS_TRAINING_FIRST_DAY, GLORYS_TRAINING_LAST_DAY)


def glorys_reanalysis_1_degree() -> xarray.Dataset:
    return open_remote_multizarr(_glorys_one_degree_monthly_paths(), parallel=False)
