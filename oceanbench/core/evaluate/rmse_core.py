import multiprocessing
from typing import Any, List

import numpy
import xarray

from oceanbench.process import get_particle_file


def get_rmse_glonet(forecast, ref, var, lead, level):
    cpu_count = multiprocessing.cpu_count()
    with multiprocessing.Pool(cpu_count) as _:
        if var == "zos":
            mask = ~numpy.isnan(forecast[var][lead]) & ~numpy.isnan(ref[var][level, lead])
            rmse = numpy.sqrt(numpy.mean((forecast[var][lead].data[mask] - ref[var][level, lead].data[mask]) ** 2))
        else:
            mask = ~numpy.isnan(forecast[var][lead, level].data) & ~numpy.isnan(ref[var][lead, level].data)
            rmse = numpy.sqrt(
                numpy.mean((forecast[var][lead, level].data[mask] - ref[var][lead, level].data[mask]) ** 2)
            )
    return rmse


def get_glonet_rmse_for_given_days(
    depthg,
    var,
    glonet_datasets: List[xarray.Dataset],
    glorys_datasets: List[xarray.Dataset],
):
    j = 0
    nweeks = 1
    aa = numpy.zeros((nweeks, 10))

    for glonet, glorys in zip(glonet_datasets, glorys_datasets):
        for i in range(0, 10):
            aa[j, i] = get_rmse_glonet(glonet, glorys, var, i, depthg)
        j = j + 1
        if j > nweeks - 1:
            break
    glonet_rmse = aa.mean(axis=0)
    return glonet_rmse


def glonet_pointwise_evaluation_core(
    glonet_datasets: List[xarray.Dataset],
    glorys_datasets: List[xarray.Dataset],
) -> numpy.ndarray[Any]:
    gnet = {"uo": [], "vo": [], "so": [], "thetao": [], "zos": []}
    variables_withouth_zos = ["uo", "vo", "so", "thetao"]
    mindepth = 0
    maxdepth = 21
    for depth in range(mindepth, maxdepth):
        print(f"{depth=}")
        for variable in variables_withouth_zos:
            gnet[variable].append(
                get_glonet_rmse_for_given_days(
                    depth,
                    variable,
                    glonet_datasets,
                    glorys_datasets,
                )
            )
        if depth < 1:
            gnet["zos"].append(
                get_glonet_rmse_for_given_days(
                    depth,
                    "zos",
                    glonet_datasets,
                    glorys_datasets,
                )
            )
    return numpy.array(gnet)


def get_euclidean_distance_core(
    first_dataset: xarray.Dataset,
    second_dataset: xarray.Dataset,
    minimum_latitude: float,
    maximum_latitude: float,
    minimum_longitude: float,
    maximum_longitude: float,
):
    gnet_traj = get_particle_file(
        first_dataset.isel(depth=0),
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
    )
    ref_traj = get_particle_file(
        second_dataset.isel(depth=0),
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
    )

    # euclidean distance
    e_d = numpy.sqrt(
        ((gnet_traj.x.data - ref_traj.x.data) * 111.32) ** 2
        + (
            111.32
            * numpy.cos(numpy.radians(gnet_traj.lat.data).reshape(1, gnet_traj.lat.shape[0], 1))
            * (gnet_traj.y.data - ref_traj.y.data)
        )
        ** 2
    )
    e_d = numpy.nanmean(e_d, axis=(1, 2))
    return e_d


def analyze_energy_cascade_core(
    glonet: xarray.Dataset,
    var,
    depth,
    spatial_resolution=None,
    small_scale_cutoff_km=100,
):

    def compute_radial_spectrum(power_spectrum):
        ny, nx = power_spectrum.shape
        y, x = numpy.ogrid[:ny, :nx]
        center = (ny // 2, nx // 2)
        r = numpy.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2).astype(int)

        radial_spectrum = numpy.zeros(r.max() + 1)
        for i in range(r.max() + 1):
            radial_spectrum[i] = power_spectrum[r == i].mean()

        return radial_spectrum

    def fill_nans(data):
        nan_mask = numpy.isnan(data)
        if nan_mask.any():
            filled_data = numpy.copy(data)
            filled_data[nan_mask] = numpy.nanmean(data)
            return filled_data
        return data

    #####
    vorticity = glonet[var][:, depth, :, :]
    n_times, _, _ = vorticity.shape

    time_spectra = []

    # process each time step
    for t in range(n_times):
        vorticity_clean = fill_nans(vorticity[t, :, :])
        vorticity_fft = numpy.fft.fft2(vorticity_clean)
        power_spectrum = numpy.abs(numpy.fft.fftshift(vorticity_fft)) ** 2
        radial_spectrum = compute_radial_spectrum(power_spectrum)
        time_spectra.append(radial_spectrum)

    time_spectra = numpy.array(time_spectra)

    # define the small-scale cutoff based on spatial resolution
    if spatial_resolution:
        # Determine the cutoff wavenumber corresponding to small_scale_cutoff_km
        grid_spacing_km = spatial_resolution * 111  # 1deg ~ 111 km
        small_scale_cutoff_index = int(small_scale_cutoff_km / grid_spacing_km)
    else:
        small_scale_cutoff_index = len(time_spectra[0]) // 2  # default: high-wavenumber half

    # compute small-scale energy fraction
    small_scale_energy = time_spectra[:, small_scale_cutoff_index:].sum(axis=1)
    total_energy = time_spectra.sum(axis=1)
    small_scale_fraction = small_scale_energy / total_energy

    return time_spectra, small_scale_fraction
