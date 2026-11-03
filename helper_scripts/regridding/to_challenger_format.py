# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import sys

import numpy
import xarray

SECONDS_PER_DAY = 86400


def to_challenger_format(dataset: xarray.Dataset) -> xarray.Dataset:
    """
    Arrange one daily forecast as an OceanBench challenger dataset.

    The first daily time becomes the ``first_day_datetime`` and the daily times become
    ``lead_day_index`` 0, 1, 2, and so on.

    Parameters
    ----------
    dataset : xarray.Dataset
        One forecast with consecutive daily values along ``time``, starting at 00:00.

    Returns
    -------
    xarray.Dataset
        The forecast with dimensions ``first_day_datetime`` and ``lead_day_index`` in place of ``time``.
    """
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


if __name__ == "__main__":
    input_path, output_path = sys.argv[1], sys.argv[2]
    with xarray.open_dataset(input_path) as dataset:
        to_challenger_format(dataset).to_netcdf(output_path)
