import pandas
import xarray

from typing import List

from oceanbench.core.derived_quantities import add_mixed_layer_depth
from oceanbench.core.derived_quantities import add_geostrophic_currents
from oceanbench.core.rmse import Variable, rmse
from oceanbench.core.references.glorys import glorys_datasets

import numpy

from oceanbench.core.process.lagrangian_analysis import get_particle_file_core


def rmse_of_variables_compared_to_glorys(
    challenger_datasets: List[xarray.Dataset],
) -> pandas.DataFrame:
    return rmse(
        challenger_datasets=challenger_datasets,
        reference_datasets=glorys_datasets(challenger_datasets),
        variables=[
            Variable.HEIGHT,
            Variable.TEMPERATURE,
            Variable.SALINITY,
            Variable.NORTHWARD_VELOCITY,
            Variable.EASTWARD_VELOCITY,
        ],
    )


def rmse_of_mixed_layer_depth_compared_to_glorys(
    challenger_datasets: List[xarray.Dataset],
) -> pandas.DataFrame:
    return rmse(
        challenger_datasets=add_mixed_layer_depth(challenger_datasets),
        reference_datasets=add_mixed_layer_depth(glorys_datasets(challenger_datasets)),
        variables=[
            Variable.MIXED_LAYER_DEPTH,
        ],
    )


def rmse_of_geostrophic_currents_compared_to_glorys(
    challenger_datasets: List[xarray.Dataset],
) -> pandas.DataFrame:
    return rmse(
        challenger_datasets=add_geostrophic_currents(challenger_datasets),
        reference_datasets=add_geostrophic_currents(glorys_datasets(challenger_datasets)),
        variables=[
            Variable.NORTHWARD_GEOSTROPHIC_VELOCITY,
            Variable.EASTWARD_GEOSTROPHIC_VELOCITY,
        ],
    )


def get_euclidean_distance_glorys_core(
    challenger_dataset: xarray.Dataset,
    minimum_latitude: float,
    maximum_latitude: float,
    minimum_longitude: float,
    maximum_longitude: float,
):
    return _get_euclidean_distance_core(
        challenger_dataset,
        glorys_datasets([challenger_dataset])[0],
        minimum_latitude,
        maximum_latitude,
        minimum_longitude,
        maximum_longitude,
    )


def _get_euclidean_distance_core(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    minimum_latitude: float,
    maximum_latitude: float,
    minimum_longitude: float,
    maximum_longitude: float,
):
    challenger_trajectory = get_particle_file_core(
        dataset=challenger_dataset.isel(depth=0),
        latzone=[minimum_latitude, maximum_latitude],
        lonzone=[minimum_longitude, maximum_longitude],
    )

    reference_trajectory = get_particle_file_core(
        dataset=reference_dataset.isel(depth=0),
        latzone=[minimum_latitude, maximum_latitude],
        lonzone=[minimum_longitude, maximum_longitude],
    )

    # euclidean distance
    e_d = numpy.sqrt(
        ((challenger_trajectory.x.data - reference_trajectory.x.data) * 111.32) ** 2
        + (
            111.32
            * numpy.cos(numpy.radians(challenger_trajectory.lat.data).reshape(1, challenger_trajectory.lat.shape[0], 1))
            * (challenger_trajectory.y.data - reference_trajectory.y.data)
        )
        ** 2
    )
    e_d = numpy.nanmean(e_d, axis=(1, 2))
    return e_d


def analyze_energy_cascade_core(
    challenger_dataset: xarray.Dataset,
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
    vorticity = challenger_dataset[var][:, depth, :, :]
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
