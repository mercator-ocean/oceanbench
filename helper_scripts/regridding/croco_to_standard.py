# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import sys

import numpy
import xarray

CROCO_TIME_ORIGIN = numpy.datetime64("2000-01-01T00:00:00", "s")
SECONDS_PER_DAY = 86400

GLO12_DEPTHS = numpy.array(
    [
        0.494025,
        1.541375,
        2.645669,
        3.819495,
        5.078224,
        6.440614,
        7.92956,
        9.572997,
        11.405,
        13.46714,
        15.81007,
        18.49556,
        21.59882,
        25.21141,
        29.44473,
        34.43415,
        40.34405,
        47.37369,
        55.76429,
        65.80727,
        77.85385,
        92.32607,
        109.7293,
        130.666,
        155.8507,
        186.1256,
        222.4752,
        266.0403,
        318.1274,
        380.213,
        453.9377,
        541.0889,
        643.5668,
        763.3331,
        902.3393,
        1062.44,
        1245.291,
        1452.251,
        1684.284,
        1941.893,
        2225.078,
        2533.336,
        2865.703,
        3220.82,
        3597.032,
        3992.484,
        4405.224,
        4833.291,
        5274.784,
        5727.917,
    ]
)

STANDARD_VARIABLE_ATTRIBUTES = {
    "thetao": {"standard_name": "sea_water_potential_temperature", "units": "degrees_C"},
    "so": {"standard_name": "sea_water_salinity", "units": "1e-3"},
    "uo": {"standard_name": "eastward_sea_water_velocity", "units": "m s-1"},
    "vo": {"standard_name": "northward_sea_water_velocity", "units": "m s-1"},
    "zos": {"standard_name": "sea_surface_height_above_geoid", "units": "m"},
}

CROCO_SOURCE_VARIABLES = {
    "thetao": ["temp"],
    "so": ["salt"],
    "uo": ["u", "v"],
    "vo": ["u", "v"],
    "zos": ["zeta"],
}


def _bare(data_array: xarray.DataArray) -> xarray.DataArray:
    return xarray.DataArray(data_array.data, dims=data_array.dims)


def _interior(data_array: xarray.DataArray) -> xarray.DataArray:
    return _bare(data_array).isel(eta_rho=slice(1, -1), xi_rho=slice(1, -1))


def _average_to_rho_points(
    staggered: xarray.DataArray, staggered_dimension: str, rho_dimension: str
) -> xarray.DataArray:
    bare = _bare(staggered)
    first = bare.isel({staggered_dimension: slice(None, -1)}).data
    second = bare.isel({staggered_dimension: slice(1, None)}).data
    return xarray.DataArray(
        0.5 * (first + second),
        dims=[rho_dimension if dimension == staggered_dimension else dimension for dimension in bare.dims],
    )


def _decoded_times(croco_dataset: xarray.Dataset) -> numpy.ndarray:
    return CROCO_TIME_ORIGIN + croco_dataset["time"].values.astype("int64").astype("timedelta64[s]")


def _water_mask(croco_dataset: xarray.Dataset) -> xarray.DataArray:
    return _interior(croco_dataset["mask_rho"]) == 1


def _level_depths(croco_dataset: xarray.Dataset) -> xarray.DataArray:
    critical_depth = float(croco_dataset["hc"])
    bathymetry = _interior(croco_dataset["h"])
    sea_surface_height = _interior(croco_dataset["zeta"])
    stretching = (critical_depth * _bare(croco_dataset["s_rho"]) + bathymetry * _bare(croco_dataset["Cs_rho"])) / (
        critical_depth + bathymetry
    )
    return -(sea_surface_height + (sea_surface_height + bathymetry) * stretching)


def _eastward_and_northward_velocities(
    croco_dataset: xarray.Dataset,
) -> tuple[xarray.DataArray, xarray.DataArray]:
    grid_angle = _interior(croco_dataset["angle"])
    x_velocity = _average_to_rho_points(croco_dataset["u"], "xi_u", "xi_rho").isel(eta_rho=slice(1, -1))
    y_velocity = _average_to_rho_points(croco_dataset["v"], "eta_v", "eta_rho").isel(xi_rho=slice(1, -1))
    eastward = x_velocity * numpy.cos(grid_angle) - y_velocity * numpy.sin(grid_angle)
    northward = y_velocity * numpy.cos(grid_angle) + x_velocity * numpy.sin(grid_angle)
    return eastward, northward


def _value_at_depth(values: numpy.ndarray, depths: numpy.ndarray, target_depth: float) -> numpy.ndarray:
    level_count = depths.shape[-1]
    shallower_count = (depths <= target_depth).sum(axis=-1, keepdims=True)
    upper_index = numpy.clip(shallower_count - 1, 0, level_count - 1)
    lower_index = numpy.clip(shallower_count, 0, level_count - 1)
    upper_depth = numpy.take_along_axis(depths, upper_index, axis=-1)
    lower_depth = numpy.take_along_axis(depths, lower_index, axis=-1)
    upper_value = numpy.take_along_axis(values, upper_index, axis=-1)
    lower_value = numpy.take_along_axis(values, lower_index, axis=-1)
    depth_span = lower_depth - upper_depth
    weight = numpy.where(depth_span > 0, (target_depth - upper_depth) / numpy.where(depth_span > 0, depth_span, 1), 0)
    interpolated = upper_value + weight * (lower_value - upper_value)
    return numpy.where(target_depth > depths[..., -1:], numpy.nan, interpolated)[..., 0]


def _interpolate_columns(values: numpy.ndarray, depths: numpy.ndarray, target_depths: numpy.ndarray) -> numpy.ndarray:
    return numpy.stack([_value_at_depth(values, depths, target_depth) for target_depth in target_depths], axis=-1)


def _at_target_depths(variable: xarray.DataArray, level_depths: xarray.DataArray) -> xarray.DataArray:
    surface_first = {"s_rho": slice(None, None, -1)}
    return xarray.apply_ufunc(
        _interpolate_columns,
        variable.isel(surface_first),
        level_depths.isel(surface_first),
        input_core_dims=[["s_rho"], ["s_rho"]],
        output_core_dims=[["depth"]],
        kwargs={"target_depths": GLO12_DEPTHS},
    ).transpose("time", "depth", "eta_rho", "xi_rho")


def _trapezoidal_daily_means(snapshots: xarray.Dataset) -> xarray.Dataset:
    snapshot_seconds = numpy.diff(snapshots["time"].values).astype("timedelta64[s]").astype("int64")
    snapshots_per_day = SECONDS_PER_DAY // int(snapshot_seconds[0])
    if not numpy.all(snapshot_seconds == snapshot_seconds[0]) or SECONDS_PER_DAY % snapshot_seconds[0] != 0:
        raise ValueError("Snapshots must be regularly spaced with a whole number of snapshots per day.")
    if (snapshots.sizes["time"] - 1) % snapshots_per_day != 0:
        raise ValueError("Snapshots must cover whole days, from 00:00 to 00:00.")
    interval_starts = snapshots.isel(time=slice(None, -1))
    interval_ends = snapshots.isel(time=slice(1, None)).assign_coords(time=interval_starts["time"])
    interval_means = 0.5 * (interval_starts + interval_ends)
    daily_means = interval_means.coarsen(time=snapshots_per_day).mean()
    return daily_means.assign_coords(time=interval_starts["time"].values[::snapshots_per_day])


def croco_to_standard_dataset(croco_dataset: xarray.Dataset) -> xarray.Dataset:
    """
    Convert a CROCO history dataset into daily means of OceanBench variables on the native CROCO grid.

    The CROCO dataset must be opened with ``decode_times=False``, use the Vtransform 2 s-coordinate,
    and hold snapshots regularly spaced from 00:00 of the first day to 00:00 after the last day.

    Land points are set to NaN, s-levels are linearly interpolated to the GLO12 depths
    (top value copied above the shallowest level, NaN below the deepest level),
    velocities are moved to rho points and rotated to eastward and northward components,
    and daily means use the trapezoidal rule over the snapshots.
    The outermost rho rows and columns are dropped.

    Parameters
    ----------
    croco_dataset : xarray.Dataset
        CROCO history dataset with temp, salt, u, v and zeta.

    Returns
    -------
    xarray.Dataset
        Daily thetao, so, uo, vo (time, depth, eta_rho, xi_rho) and zos (time, eta_rho, xi_rho),
        with 2-D latitude and longitude coordinates.
    """
    water_mask = _water_mask(croco_dataset)
    level_depths = _level_depths(croco_dataset)
    eastward, northward = _eastward_and_northward_velocities(croco_dataset)
    native_level_variables = {
        "thetao": _interior(croco_dataset["temp"]),
        "so": _interior(croco_dataset["salt"]),
        "uo": eastward,
        "vo": northward,
    }
    snapshots = xarray.Dataset(
        {
            **{
                name: _at_target_depths(variable.where(water_mask), level_depths)
                for name, variable in native_level_variables.items()
            },
            "zos": _interior(croco_dataset["zeta"]).where(water_mask),
        },
        coords={
            "time": _decoded_times(croco_dataset),
            "depth": ("depth", GLO12_DEPTHS, {"standard_name": "depth", "units": "m", "positive": "down"}),
            "latitude": (
                ("eta_rho", "xi_rho"),
                _interior(croco_dataset["lat_rho"]).values,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "longitude": (
                ("eta_rho", "xi_rho"),
                _interior(croco_dataset["lon_rho"]).values,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
        },
    )
    daily_means = _trapezoidal_daily_means(snapshots)
    return daily_means.assign(
        {
            name: daily_means[name]
            .astype(numpy.result_type(*[croco_dataset[source].dtype for source in CROCO_SOURCE_VARIABLES[name]]))
            .assign_attrs(attributes)
            for name, attributes in STANDARD_VARIABLE_ATTRIBUTES.items()
        }
    )


if __name__ == "__main__":
    input_path, output_path = sys.argv[1], sys.argv[2]
    with xarray.open_dataset(input_path, decode_times=False) as croco_dataset:
        croco_to_standard_dataset(croco_dataset).to_netcdf(output_path)
