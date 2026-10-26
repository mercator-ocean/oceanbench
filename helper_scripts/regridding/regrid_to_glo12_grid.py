# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import sys

import numpy
import xarray
from pyresample.bilinear import NumpyBilinearResampler
from pyresample.geometry import AreaDefinition, SwathDefinition

GLO12_FIRST_LONGITUDE = -180.0
GLO12_FIRST_LATITUDE = -80.0
GLO12_GRID_SPACING = 1.0 / 12.0
RADIUS_OF_INFLUENCE_METRES = 50e3
NEWTON_ITERATIONS = 10
FRACTIONAL_POSITION_TOLERANCE = 1e-6


def _glo12_indices_covering(coordinates: numpy.ndarray, first_coordinate: float) -> numpy.ndarray:
    positions = (coordinates - first_coordinate) / GLO12_GRID_SPACING
    return numpy.arange(int(numpy.ceil(numpy.nanmin(positions))), int(numpy.floor(numpy.nanmax(positions))) + 1)


def _glo12_coordinates(indices: numpy.ndarray, first_coordinate: float) -> numpy.ndarray:
    return numpy.float32(first_coordinate) + indices.astype(numpy.float32) * numpy.float32(GLO12_GRID_SPACING)


def _target_area(target_longitudes: numpy.ndarray, target_latitudes: numpy.ndarray) -> AreaDefinition:
    half_spacing = GLO12_GRID_SPACING / 2
    return AreaDefinition(
        "glo12_subset",
        "GLO12 1/12 degree grid subset",
        "longlat",
        {"proj": "longlat", "datum": "WGS84"},
        target_longitudes.size,
        target_latitudes.size,
        (
            target_longitudes[0] - half_spacing,
            target_latitudes[0] - half_spacing,
            target_longitudes[-1] + half_spacing,
            target_latitudes[-1] + half_spacing,
        ),
    )


def _longitudes_relative_to(longitudes: numpy.ndarray, reference_longitudes: numpy.ndarray) -> numpy.ndarray:
    return (longitudes - reference_longitudes + 180.0) % 360.0 - 180.0


def _bilinear_position(
    corner_x: numpy.ndarray, corner_y: numpy.ndarray, s: numpy.ndarray, t: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray]:
    corner_weights = numpy.stack([(1 - s) * (1 - t), s * (1 - t), (1 - s) * t, s * t], axis=-1)
    return (corner_weights * corner_x).sum(axis=-1), (corner_weights * corner_y).sum(axis=-1)


def _newton_step(
    corner_x: numpy.ndarray, corner_y: numpy.ndarray, s: numpy.ndarray, t: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray]:
    x_residual, y_residual = _bilinear_position(corner_x, corner_y, s, t)
    corner_1, corner_2, corner_3, corner_4 = ((corner_x[:, index], corner_y[:, index]) for index in range(4))
    x_by_s = (1 - t) * (corner_2[0] - corner_1[0]) + t * (corner_4[0] - corner_3[0])
    y_by_s = (1 - t) * (corner_2[1] - corner_1[1]) + t * (corner_4[1] - corner_3[1])
    x_by_t = (1 - s) * (corner_3[0] - corner_1[0]) + s * (corner_4[0] - corner_2[0])
    y_by_t = (1 - s) * (corner_3[1] - corner_1[1]) + s * (corner_4[1] - corner_2[1])
    determinant = x_by_s * y_by_t - x_by_t * y_by_s
    safe_determinant = numpy.where(determinant != 0, determinant, numpy.nan)
    return (
        s - (y_by_t * x_residual - x_by_t * y_residual) / safe_determinant,
        t - (x_by_s * y_residual - y_by_s * x_residual) / safe_determinant,
    )


def _fractional_positions(corner_x: numpy.ndarray, corner_y: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    starting_guess = numpy.full(corner_x.shape[0], 0.5)
    s, t = starting_guess, starting_guess
    for _ in range(NEWTON_ITERATIONS):
        s, t = _newton_step(corner_x, corner_y, s, t)
    outside = (
        (s < -FRACTIONAL_POSITION_TOLERANCE)
        | (s > 1 + FRACTIONAL_POSITION_TOLERANCE)
        | (t < -FRACTIONAL_POSITION_TOLERANCE)
        | (t > 1 + FRACTIONAL_POSITION_TOLERANCE)
    )
    return (
        numpy.where(outside, numpy.nan, numpy.clip(s, 0, 1)),
        numpy.where(outside, numpy.nan, numpy.clip(t, 0, 1)),
    )


def _bilinear_lookup(
    source_longitudes: numpy.ndarray,
    source_latitudes: numpy.ndarray,
    target_longitudes: numpy.ndarray,
    target_latitudes: numpy.ndarray,
) -> dict[str, numpy.ndarray]:
    flat_longitudes = source_longitudes.astype(numpy.float64).ravel()
    flat_latitudes = source_latitudes.astype(numpy.float64).ravel()
    resampler = NumpyBilinearResampler(
        SwathDefinition(source_longitudes.astype(numpy.float64), source_latitudes.astype(numpy.float64)),
        _target_area(target_longitudes, target_latitudes),
        RADIUS_OF_INFLUENCE_METRES,
    )
    resampler.get_bil_info()
    corner_indices = resampler.slices_y * source_longitudes.shape[1] + resampler.slices_x
    north_first_longitudes, north_first_latitudes = (
        coordinates.astype(numpy.float64).ravel()
        for coordinates in numpy.meshgrid(target_longitudes, target_latitudes[::-1])
    )
    corner_x = _longitudes_relative_to(flat_longitudes[corner_indices], north_first_longitudes[:, numpy.newaxis])
    corner_y = flat_latitudes[corner_indices] - north_first_latitudes[:, numpy.newaxis]
    s, t = (position[:, numpy.newaxis] for position in _fractional_positions(corner_x, corner_y))
    corner_weights = numpy.concatenate([(1 - s) * (1 - t), s * (1 - t), (1 - s) * t, s * t], axis=1)
    return {
        "corner_indices": corner_indices,
        "corner_weights": numpy.where(resampler.mask_slices | numpy.isnan(corner_weights), 0.0, corner_weights),
    }


def _apply_lookup(values: numpy.ndarray, corner_indices: numpy.ndarray, corner_weights: numpy.ndarray) -> numpy.ndarray:
    corner_values = values.reshape(values.shape[:-2] + (-1,))[..., corner_indices]
    weights = numpy.where(numpy.isnan(corner_values), 0.0, corner_weights)
    weight_sum = weights.sum(axis=-1)
    weighted_sum = numpy.where(weights > 0, weights * corner_values, 0.0).sum(axis=-1)
    return numpy.where(weight_sum > 0, weighted_sum / numpy.where(weight_sum > 0, weight_sum, 1.0), numpy.nan)


def _regridded_variable(
    variable: xarray.DataArray,
    lookup: dict[str, numpy.ndarray],
    target_latitudes: numpy.ndarray,
    target_longitudes: numpy.ndarray,
) -> xarray.DataArray:
    flat_target = _apply_lookup(variable.values, lookup["corner_indices"], lookup["corner_weights"])
    north_first = flat_target.reshape(flat_target.shape[:-1] + (target_latitudes.size, target_longitudes.size))
    return xarray.DataArray(
        north_first[..., ::-1, :].astype(variable.dtype),
        dims=variable.dims[:-2] + ("latitude", "longitude"),
        attrs=variable.attrs,
    )


def regrid_to_glo12_grid(dataset: xarray.Dataset) -> xarray.Dataset:
    """
    Regrid a curvilinear dataset onto the GLO12 1/12 degree latitude-longitude grid.

    Bilinear weights come from pyresample on the sphere.
    Source corners that are NaN (land, below the seabed or outside the domain) are dropped
    and the remaining weights renormalised, so a target point is NaN only when all four corners are.
    The target grid covers the GLO12 points inside the source latitude and longitude range.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset whose variables end with the two horizontal dimensions,
        with 2-D latitude and longitude coordinates on those dimensions.

    Returns
    -------
    xarray.Dataset
        The same variables on 1-D ascending latitude and longitude coordinates of the GLO12 grid.
    """
    source_longitudes = dataset["longitude"].values
    source_latitudes = dataset["latitude"].values
    target_longitudes = _glo12_coordinates(
        _glo12_indices_covering(source_longitudes, GLO12_FIRST_LONGITUDE), GLO12_FIRST_LONGITUDE
    )
    target_latitudes = _glo12_coordinates(
        _glo12_indices_covering(source_latitudes, GLO12_FIRST_LATITUDE), GLO12_FIRST_LATITUDE
    )
    lookup = _bilinear_lookup(source_longitudes, source_latitudes, target_longitudes, target_latitudes)
    return xarray.Dataset(
        {
            name: _regridded_variable(variable, lookup, target_latitudes, target_longitudes)
            for name, variable in dataset.data_vars.items()
        },
        coords={
            **{name: dataset[name] for name in ("time", "depth") if name in dataset.coords},
            "latitude": (
                "latitude",
                target_latitudes,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "longitude": (
                "longitude",
                target_longitudes,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
        },
    )


if __name__ == "__main__":
    input_path, output_path = sys.argv[1], sys.argv[2]
    with xarray.open_dataset(input_path) as dataset:
        regrid_to_glo12_grid(dataset).to_netcdf(output_path)
