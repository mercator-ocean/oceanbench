# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# Open persistence baseline 1 degree interpolated forecasts with xarray
import xarray
import oceanbench

challenger_dataset: xarray.Dataset = oceanbench.datasets.challenger.persistence_1_degree()

challenger_dataset
