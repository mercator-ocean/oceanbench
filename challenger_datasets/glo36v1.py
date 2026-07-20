# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import xarray
import oceanbench

challenger_dataset: xarray.Dataset = oceanbench.datasets.challenger.glo36v1()

challenger_dataset
