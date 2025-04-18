import pandas
import xarray

from typing import List

from oceanbench.core.derived_quantities import add_mixed_layer_depth
from oceanbench.core.derived_quantities import add_geostrophic_currents
from oceanbench.core.rmsd import Variable, rmsd
from oceanbench.core.references.glorys import glorys_datasets

import numpy

from oceanbench.core.lagrangian_trajectory import (
    Zone,
    deviation_of_lagrangian_trajectories,
)


def rmsd_of_variables_compared_to_glorys(
    challenger_datasets: List[xarray.Dataset],
) -> pandas.DataFrame:
    return rmsd(
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


def rmsd_of_mixed_layer_depth_compared_to_glorys(
    challenger_datasets: List[xarray.Dataset],
) -> pandas.DataFrame:
    return rmsd(
        challenger_datasets=add_mixed_layer_depth(challenger_datasets),
        reference_datasets=add_mixed_layer_depth(glorys_datasets(challenger_datasets)),
        variables=[
            Variable.MIXED_LAYER_DEPTH,
        ],
    )


def rmsd_of_geostrophic_currents_compared_to_glorys(
    challenger_datasets: List[xarray.Dataset],
) -> pandas.DataFrame:
    return rmsd(
        challenger_datasets=add_geostrophic_currents(challenger_datasets),
        reference_datasets=add_geostrophic_currents(glorys_datasets(challenger_datasets)),
        variables=[
            Variable.NORTHWARD_GEOSTROPHIC_VELOCITY,
            Variable.EASTWARD_GEOSTROPHIC_VELOCITY,
        ],
    )


def deviation_of_lagrangian_trajectories_compared_to_glorys(
    challenger_datasets: List[xarray.Dataset],
) -> pandas.DataFrame:
    return deviation_of_lagrangian_trajectories(
        challenger_datasets=challenger_datasets,
        reference_datasets=glorys_datasets(challenger_datasets),
        zone=Zone.SMALL_ATLANTIC_NEWYORK_TO_NOUADHIBOU,
    )


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
