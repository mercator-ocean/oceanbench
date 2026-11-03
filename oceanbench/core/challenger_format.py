# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

SECONDS_PER_DAY = 86400


def to_challenger_format(dataset: xarray.Dataset) -> xarray.Dataset:
    times = dataset["time"].values
    day_steps = numpy.diff(times).astype("timedelta64[s]").astype("int64")
    if not numpy.all(day_steps == SECONDS_PER_DAY):
        raise ValueError("Forecast times must be consecutive days.")
    if times[0] != times[0].astype("datetime64[D]"):
        raise ValueError("The first forecast time must be at 00:00.")
    return (
        dataset.rename({"time": "lead_day_index"})
        .assign_coords(lead_day_index=numpy.arange(times.size))
        .expand_dims(first_day_datetime=[times[0]])
    )
