from datetime import datetime
from functools import partial
import multiprocessing
from typing import List, Optional

import numpy
import xarray
import pandas

from oceanbench.core.references.glorys import glorys_datasets
from oceanbench.core.process.lagrangian_analysis import get_particle_file_core
from itertools import product

DEPTH_STANDARD_NAME = "depth"
TIME_STANDARD_NAME = "time"


def _get_variable_name_from_standard_name(dataset: xarray.Dataset, standard_name: str) -> Optional[str]:
    for variable_name in dataset.variables:
        if hasattr(dataset[variable_name], "standard_name") and dataset[variable_name].standard_name == standard_name:
            return str(variable_name)
    return None


def _rmse(data, reference_data):
    mask = ~numpy.isnan(data) & ~numpy.isnan(reference_data)
    rmse = numpy.sqrt(numpy.mean((data[mask] - reference_data[mask]) ** 2))
    return rmse


def _select_variable_day_and_depth(
    dataset: xarray.Dataset,
    variable_name: str,
    depth_level_meter: float,
    lead_day: int,
) -> xarray.DataArray:
    depth_name = _get_variable_name_from_standard_name(dataset, DEPTH_STANDARD_NAME)
    time_name = _get_variable_name_from_standard_name(dataset, TIME_STANDARD_NAME)
    try:
        new_dataset = dataset[variable_name].isel({time_name: lead_day})
        return (
            new_dataset.sel({depth_name: depth_level_meter})
            if depth_name in dataset[variable_name].coords
            else new_dataset
        )
    except Exception as exception:
        start_datetime = datetime.fromisoformat(str(dataset[time_name][0].values))
        details = (
            f"start_datetime={start_datetime}, variable={variable_name}, depth={depth_level_meter}, lead_day={lead_day}"
        )
        raise Exception(f"Could not select data: {details}") from exception


def _get_rmse(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variable_name: str,
    depth_level_meter: float,
    lead_day: int,
) -> float:
    cpu_count = multiprocessing.cpu_count()
    with multiprocessing.Pool(cpu_count) as _:
        challenger_dataarray = _select_variable_day_and_depth(
            challenger_dataset, variable_name, depth_level_meter, lead_day
        )
        reference_dataarray = _select_variable_day_and_depth(
            reference_dataset, variable_name, depth_level_meter, lead_day
        )
        return _rmse(challenger_dataarray.data, reference_dataarray.data)


LEAD_DAYS_COUNT = 10


def get_rmse_for_all_lead_days(
    dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variable_name: str,
    depth_level_meter: float,
) -> list[float]:
    return list(
        map(
            partial(
                _get_rmse,
                dataset,
                reference_dataset,
                variable_name,
                depth_level_meter,
            ),
            range(LEAD_DAYS_COUNT),
        )
    )


def _compute_rmse(
    datasets: List[xarray.Dataset],
    reference_datasets: List[xarray.Dataset],
    variable_name: str,
    depth_level_meter: float,
) -> numpy.ndarray[float]:

    all_rmse = numpy.array(
        [
            get_rmse_for_all_lead_days(dataset, reference_dataset, variable_name, depth_level_meter)
            for dataset, reference_dataset in zip(datasets, reference_datasets)
        ]
    )
    return all_rmse.mean(axis=0)


def pointwise_evaluation_glorys_core(
    challenger_datasets: List[xarray.Dataset],
) -> pandas.DataFrame:
    return _pointwise_evaluation_core(challenger_datasets, glorys_datasets(challenger_datasets))


def _lead_day_labels(day_count) -> list[str]:
    return list(
        map(
            lambda day_index: f"Lead day {day_index}",
            range(1, day_count + 1),
        )
    )


VARIABLE_LABELS = {
    "zos": "height",
    "thetao": "temperature",
    "so": "salinity",
    "vo": "northward velocity",
    "uo": "eastward velocity",
}

DEPTH_LABELS = {
    4.940250e-01: "Surface",
    4.737369e01: "50m",
    2.224752e02: "200m",
    5.410889e02: "550m",
}


def _variale_depth_label(variable_name: str, depth_level_meter: float) -> str:
    return f"{DEPTH_LABELS[depth_level_meter]} {VARIABLE_LABELS[variable_name]}"


def _has_depths(dataset: xarray.Dataset, variable_name: str) -> bool:
    return _get_variable_name_from_standard_name(dataset, DEPTH_STANDARD_NAME) in dataset[variable_name].coords


def _is_surface(depth_level_meter: float) -> bool:
    return depth_level_meter == list(DEPTH_LABELS.keys())[0]


def _variable_and_depth_combinations(
    dataset: xarray.Dataset,
) -> list[tuple[str, float]]:
    return list(
        (variable_name, depth_level_meter)
        for (depth_level_meter, variable_name) in product(DEPTH_LABELS.keys(), VARIABLE_LABELS.keys())
        if (_has_depths(dataset, variable_name) or _is_surface(depth_level_meter))
    )


def _pointwise_evaluation_core(
    challenger_datasets: List[xarray.Dataset],
    reference_datasets: List[xarray.Dataset],
) -> pandas.DataFrame:
    all_combinations = _variable_and_depth_combinations(challenger_datasets[0])
    scores = {
        _variale_depth_label(variable_name, depth_level_meter): list(
            _compute_rmse(
                challenger_datasets,
                reference_datasets,
                variable_name,
                depth_level_meter,
            )
        )
        for (variable_name, depth_level_meter) in all_combinations
    }
    score_dataframe = pandas.DataFrame(scores)
    score_dataframe.index = _lead_day_labels(len(score_dataframe))
    return score_dataframe.T


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
