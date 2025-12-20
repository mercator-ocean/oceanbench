# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# Open XiHe forecasts with xarray
import xarray
import oceanbench

challenger_dataset: xarray.Dataset = oceanbench.datasets.challenger.xihe()

challenger_dataset
