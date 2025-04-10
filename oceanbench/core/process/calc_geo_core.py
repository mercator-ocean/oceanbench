import numpy
import xarray


def calc_geo_core(dataset: xarray.Dataset, var: str, lead: int) -> xarray.Dataset:
    ssh = dataset[var].values
    lat = dataset["lat"].values
    lon = dataset["lon"].values

    # to radian
    lat_r = numpy.deg2rad(lat)

    # coriolis
    omega = 7.2921e-5
    f = 2 * omega * numpy.sin(lat_r)
    R = 6371000

    # Compute grid spacing
    dx = numpy.gradient(lon) * (numpy.pi / 180) * R * numpy.cos(lat_r[:, numpy.newaxis])
    dy = numpy.gradient(lat)[:, numpy.newaxis] * (numpy.pi / 180) * R

    dssh_dx = numpy.gradient(ssh, axis=-1) / dx
    dssh_dy = numpy.gradient(ssh, axis=-2) / dy

    g = 9.81  # gravity
    u_geo = -g / f[:, numpy.newaxis] * dssh_dy
    v_geo = g / f[:, numpy.newaxis] * dssh_dx

    dataset["u_geo"] = (("time","lat", "lon"), u_geo)
    dataset["v_geo"] = (("time","lat", "lon"), v_geo)

    return dataset
