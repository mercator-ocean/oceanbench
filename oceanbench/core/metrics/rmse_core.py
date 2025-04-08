from functools import partial
import multiprocessing
from typing import List

import numpy
import xarray
import pandas

from oceanbench.core.references.glorys import glorys_datasets
from oceanbench.core.process.lagrangian_analysis import get_particle_file_core
from oceanbench.core.utils.score import Score
from IPython.display import display, HTML


def _get_rmse(forecast, ref, var, lead, level):
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


def _get_rmse_for_given_days(
    depthg,
    var,
    datasets: List[xarray.Dataset],
    glorys_datasets: List[xarray.Dataset],
) -> numpy.ndarray:
    j = 0
    nweeks = 1
    aa = numpy.zeros((nweeks, 10))

    for dataset, glorys_dataset in zip(datasets, glorys_datasets):
        for i in range(0, 10):
            aa[j, i] = _get_rmse(dataset, glorys_dataset, var, i, depthg)
        j = j + 1
        if j > nweeks - 1:
            break
    rmse = aa.mean(axis=0)
    return rmse


def pointwise_evaluation_glorys_core(
    candidate_datasets: List[xarray.Dataset],
    display_html: bool,
) -> Score:
    variable_evaluations = _pointwise_evaluation_core(candidate_datasets, glorys_datasets(candidate_datasets))
    if display_html:
        _display_html(variable_evaluations)
    return variable_evaluations


def _display_variable_html(
    variable_evaluations: dict[str, list[numpy.ndarray]],
    variable_name: str,
):

    variable_evaluation = numpy.array(variable_evaluations[variable_name])
    display(HTML(f'<h1 style="color:red; text-align:center;">Surface {variable_name} score</h1>'))
    df = pandas.DataFrame(
        [
            ["Lead Day " + str(i + 1) for i in range(10)],
            variable_evaluation[0, :],
        ]
    )
    df.index = ["", "Score"]
    df.style.set_properties(**{"border": "1px solid black", "text-align": "center"})
    if variable_name != "zos":
        display(HTML(f'<h1 style="color:red; text-align:center;">50m {variable_name} score</h1>'))
        df = pandas.DataFrame(
            [
                ["Lead Day " + str(i + 1) for i in range(10)],
                variable_evaluation[1, :],
            ]
        )
        df.index = ["", "Score"]
        df.style.set_properties(**{"border": "1px solid black", "text-align": "center"})


def _display_html(variable_evaluations: dict[str, list[numpy.ndarray]]):
    list(
        map(
            partial(_display_variable_html, variable_evaluations),
            ["uo", "vo", "thetao", "so", "zos"],
        )
    )


def _pointwise_evaluation_core(
    candidate_datasets: List[xarray.Dataset],
    reference_datasets: List[xarray.Dataset],
) -> dict[str, list[numpy.ndarray]]:
    rmse_by_variable: dict[str, list[numpy.ndarray]] = {
        "uo": [],
        "vo": [],
        "so": [],
        "thetao": [],
        "zos": [],
    }
    variables_withouth_zos = ["uo", "vo", "so", "thetao"]
    mindepth = 0
    maxdepth = 1
    for depth in range(mindepth, maxdepth):
        print(f"{depth=}")
        for variable in variables_withouth_zos:
            rmse_by_variable[variable].append(
                _get_rmse_for_given_days(
                    depth,
                    variable,
                    candidate_datasets,
                    reference_datasets,
                )
            )
        if depth < 1:
            rmse_by_variable["zos"].append(
                _get_rmse_for_given_days(
                    depth,
                    "zos",
                    candidate_datasets,
                    reference_datasets,
                )
            )
    return rmse_by_variable


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
