import multiprocessing
from typing import List

import numpy
import xarray
import pandas

from oceanbench.core.references.glorys import glorys_datasets
from oceanbench.core.process.lagrangian_analysis import get_particle_file_core
from itertools import product


def _get_rmse(forecast, ref, var, lead, depth_level):
    cpu_count = multiprocessing.cpu_count()
    with multiprocessing.Pool(cpu_count) as _:
        if var == "zos":
            mask = ~numpy.isnan(forecast[var][lead]) & ~numpy.isnan(ref[var][depth_level, lead])
            rmse = numpy.sqrt(
                numpy.mean((forecast[var][lead].data[mask] - ref[var][depth_level, lead].data[mask]) ** 2)
            )
        else:
            mask = ~numpy.isnan(forecast[var][lead, depth_level].data) & ~numpy.isnan(ref[var][lead, depth_level].data)
            rmse = numpy.sqrt(
                numpy.mean((forecast[var][lead, depth_level].data[mask] - ref[var][lead, depth_level].data[mask]) ** 2)
            )
    return rmse


def _compute_rmse(
    datasets: List[xarray.Dataset],
    glorys_datasets: List[xarray.Dataset],
    variable_name: str,
    depth_level: int,
) -> numpy.ndarray:
    j = 0
    nweeks = 1
    aa = numpy.zeros((nweeks, 10))

    for dataset, glorys_dataset in zip(datasets, glorys_datasets):
        for i in range(0, 10):
            aa[j, i] = _get_rmse(dataset, glorys_dataset, variable_name, i, depth_level)
        j = j + 1
        if j > nweeks - 1:
            break
    rmse = aa.mean(axis=0)
    return rmse


def pointwise_evaluation_glorys_core(
    candidate_datasets: List[xarray.Dataset],
) -> pandas.DataFrame:
    return _pointwise_evaluation_core(candidate_datasets, glorys_datasets(candidate_datasets))


def _lead_day_labels(day_count) -> list[str]:
    return list(
        map(
            lambda day_index: f"Lead day {day_index}",
            range(1, day_count + 1),
        )
    )


VARIABLE_LABELS = {
    "thetao": "temperature",
    "so": "salinity",
    "zos": "height",
    "vo": "northward velocity",
    "uo": "eastward velocity",
}

DEPTH_LABELS = {
    0: "Surface",
    1: "50m",
}


def _variale_depth_label(variable_name: str, depth_level: int) -> str:
    return f"{DEPTH_LABELS[depth_level]} {VARIABLE_LABELS[variable_name]}"


def _pointwise_evaluation_core(
    candidate_datasets: List[xarray.Dataset],
    reference_datasets: List[xarray.Dataset],
) -> pandas.DataFrame:
    all_combinations = list(product(VARIABLE_LABELS.keys(), DEPTH_LABELS.keys()))
    scores = {
        _variale_depth_label(variable_name, depth_level): list(
            _compute_rmse(
                candidate_datasets,
                reference_datasets,
                variable_name,
                depth_level,
            )
        )
        for (variable_name, depth_level) in all_combinations
    }
    score_dataframe = pandas.DataFrame(scores)
    score_dataframe.index = _lead_day_labels(len(score_dataframe))
    return score_dataframe.T


def get_euclidean_distance_glorys_core(
    candidate_dataset: xarray.Dataset,
    minimum_latitude: float,
    maximum_latitude: float,
    minimum_longitude: float,
    maximum_longitude: float,
):
    return _get_euclidean_distance_core(
        candidate_dataset,
        glorys_datasets([candidate_dataset])[0],
        minimum_latitude,
        maximum_latitude,
        minimum_longitude,
        maximum_longitude,
    )


def _get_euclidean_distance_core(
    candidate_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    minimum_latitude: float,
    maximum_latitude: float,
    minimum_longitude: float,
    maximum_longitude: float,
):
    candidate_trajectory = get_particle_file_core(
        dataset=candidate_dataset.isel(depth=0),
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
        ((candidate_trajectory.x.data - reference_trajectory.x.data) * 111.32) ** 2
        + (
            111.32
            * numpy.cos(numpy.radians(candidate_trajectory.lat.data).reshape(1, candidate_trajectory.lat.shape[0], 1))
            * (candidate_trajectory.y.data - reference_trajectory.y.data)
        )
        ** 2
    )
    e_d = numpy.nanmean(e_d, axis=(1, 2))
    return e_d


def analyze_energy_cascade_core(
    candidate_dataset: xarray.Dataset,
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
    vorticity = candidate_dataset[var][:, depth, :, :]
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
