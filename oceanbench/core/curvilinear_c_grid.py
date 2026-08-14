# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Where the velocity points of a NEMO Arakawa C-grid actually sit.

A NEMO ocean model carries its tracers at the centre of a cell and its velocities on the
cell faces: the zonal velocity at the middle of the eastern face and the meridional velocity
at the middle of the northern face, each half a cell from the tracer point. On a regular grid
that half cell is a small enough offset to ignore; on the tripolar grid of GloEns the cells
turn with the fold, and the offset is a genuine displacement in both latitude and longitude.

The GloEns 3D velocity stores publish latitude and longitude arrays that are entirely missing
values, so the velocity positions cannot be read and have to be reconstructed from the tracer
grid under the NEMO convention. The convention itself was validated on the data during the
spectra study, where the reconstructed positions gave the maximum correlation with the GLORYS
surface currents.

The reconstruction is one half step along the grid index the point is staggered on:

    U(j, i) is halfway between T(j, i) and T(j, i + 1), the next cell along the row
    V(j, i) is halfway between T(j, i) and T(j + 1, i), the next cell along the column

The last row and the last column have no next cell, so the last interior step is repeated
there. That is exact wherever the grid spacing is locally constant, which is everywhere in
the Mercator band, and approximate on the last row, which on a tripolar grid runs along the
fold where the two halves of the northern boundary meet and the local step stops describing
the geometry. Those rows are ocean only in the Arctic and the displacement they carry is a
fraction of a cell, so a target point that would take a badly placed cell is past the
neighbour distance cutoff of the sampling and is dropped rather than sampled.
"""

import numpy

GRID_TYPE_TRACER = "T"
GRID_TYPE_ZONAL_VELOCITY = "U"
GRID_TYPE_MERIDIONAL_VELOCITY = "V"

#: Store variable name to the C-grid point its field is carried on.
#:
#: Only the velocity components are staggered, so every other field is a tracer point field
#: and anything not named here is treated as one. Both the store names of the GloEns stores
#: and the standard names the library renames them to are listed, so a field can be resolved
#: before or after the rename.
GRID_TYPE_BY_VARIABLE_NAME: dict[str, str] = {
    "uo": GRID_TYPE_ZONAL_VELOCITY,
    "eastward_sea_water_velocity": GRID_TYPE_ZONAL_VELOCITY,
    "vo": GRID_TYPE_MERIDIONAL_VELOCITY,
    "northward_sea_water_velocity": GRID_TYPE_MERIDIONAL_VELOCITY,
}


def grid_type_of_variable(variable_name: str) -> str:
    """The C-grid point ``variable_name`` is carried on, tracer point unless it is a velocity."""
    return GRID_TYPE_BY_VARIABLE_NAME.get(variable_name, GRID_TYPE_TRACER)


def _steps_along(values: numpy.ndarray, axis: int) -> numpy.ndarray:
    """First differences along ``axis``, with the last interior step repeated at the end."""
    steps = numpy.diff(values, axis=axis)
    return numpy.concatenate([steps, numpy.take(steps, [-1], axis=axis)], axis=axis)


def _longitude_steps_along(longitude: numpy.ndarray, axis: int) -> numpy.ndarray:
    """First differences of a longitude, taken as angles rather than as numbers.

    A row that crosses the date line steps from 179.9 to -179.9, which is a tenth of a degree
    east and not 359.8 degrees west. Wrapping the difference into (-180, 180] before it is
    halved keeps the midpoint of the two cells between them.
    """
    return (_steps_along(longitude, axis) + 180.0) % 360.0 - 180.0


def _wrapped_longitude(longitude: numpy.ndarray) -> numpy.ndarray:
    return (longitude + 180.0) % 360.0 - 180.0


def c_grid_positions(
    tracer_latitude: numpy.ndarray,
    tracer_longitude: numpy.ndarray,
    grid_type: str,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """The ``(latitude, longitude)`` of the requested C-grid point, from the tracer grid.

    Both inputs are the two-dimensional ``(y, x)`` tracer arrays of the model grid and both
    outputs have their shape. The tracer point is returned unchanged.
    """
    latitude = numpy.asarray(tracer_latitude, dtype="float64")
    longitude = numpy.asarray(tracer_longitude, dtype="float64")
    if latitude.ndim != 2 or longitude.ndim != 2 or latitude.shape != longitude.shape:
        raise ValueError(
            "the tracer grid must be two two-dimensional arrays of the same shape, "
            f"got latitude {latitude.shape} and longitude {longitude.shape}"
        )
    if grid_type == GRID_TYPE_TRACER:
        return latitude, longitude
    if grid_type == GRID_TYPE_ZONAL_VELOCITY:
        axis = 1
    elif grid_type == GRID_TYPE_MERIDIONAL_VELOCITY:
        axis = 0
    else:
        raise ValueError(
            f"unknown C-grid point '{grid_type}', expected one of "
            f"{GRID_TYPE_TRACER}, {GRID_TYPE_ZONAL_VELOCITY}, {GRID_TYPE_MERIDIONAL_VELOCITY}"
        )
    return (
        latitude + 0.5 * _steps_along(latitude, axis),
        _wrapped_longitude(longitude + 0.5 * _longitude_steps_along(longitude, axis)),
    )
