# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# Open GLORYS 1/4 degree reanalysis as a reference challenger with xarray
import xarray
import oceanbench

challenger_dataset: xarray.Dataset = oceanbench.datasets.challenger.glorys()
# The quarter-degree GLORYS files name their horizontal dimensions lat and lon, the other challengers latitude and longitude
challenger_dataset = challenger_dataset.rename(
    {name: standard for name, standard in {"lat": "latitude", "lon": "longitude"}.items() if name in challenger_dataset.dims}
)

challenger_dataset
