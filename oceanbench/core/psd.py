# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from metpy.calc import lat_lon_grid_deltas
from metpy.units import units
import numpy
import pandas
import xarray
import xrft

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.lead_day_utils import lead_day_labels

DEFAULT_FILL_VALUE = 0.0
DEFAULT_REGIONAL_SCALE_WAVELENGTH_THRESHOLD_KM = 500.0
DEFAULT_LARGE_SCALE_WAVELENGTH_THRESHOLD_KM = 2000.0
DEFAULT_NEAR_GRID_SCALE_BAND_NAME = "Near-grid scale"
DEFAULT_REGIONAL_SCALE_BAND_NAME = "Regional scale"
DEFAULT_LARGE_SCALE_BAND_NAME = "Large scale"


def _as_variable_key(variable: Variable | str) -> str:
    return variable.key() if isinstance(variable, Variable) else variable


def _rename_psd_dimensions(data_array: xarray.DataArray) -> xarray.DataArray:
    rename_map = {}
    if Dimension.LATITUDE.key() in data_array.dims or Dimension.LATITUDE.key() in data_array.coords:
        rename_map[Dimension.LATITUDE.key()] = "lat"
    if Dimension.LONGITUDE.key() in data_array.dims or Dimension.LONGITUDE.key() in data_array.coords:
        rename_map[Dimension.LONGITUDE.key()] = "lon"
    if Dimension.LEAD_DAY_INDEX.key() in data_array.dims or Dimension.LEAD_DAY_INDEX.key() in data_array.coords:
        rename_map[Dimension.LEAD_DAY_INDEX.key()] = "time"
    return data_array.rename(rename_map)


def _select_first_day_and_depth(
    data_array: xarray.DataArray,
    first_day_index: int,
    depth_selector: int | float | None,
) -> xarray.DataArray:
    if Dimension.FIRST_DAY_DATETIME.key() in data_array.dims:
        data_array = data_array.isel({Dimension.FIRST_DAY_DATETIME.key(): first_day_index})
    if Dimension.DEPTH.key() in data_array.dims:
        if depth_selector is None:
            data_array = data_array.isel({Dimension.DEPTH.key(): 0})
        elif isinstance(depth_selector, int):
            data_array = data_array.isel({Dimension.DEPTH.key(): depth_selector})
        else:
            data_array = data_array.sel({Dimension.DEPTH.key(): depth_selector}, method="nearest")
    return data_array


def _wrap_longitudes(data_array: xarray.DataArray) -> xarray.DataArray:
    wrapped_longitudes = ((data_array["lon"] + 180.0) % 360.0) - 180.0
    return data_array.assign_coords(lon=wrapped_longitudes).sortby("lon")


def _regularize_rectilinear_grid(data_array: xarray.DataArray) -> xarray.DataArray:
    if data_array.sizes["lat"] < 2 or data_array.sizes["lon"] < 2:
        return data_array
    latitude_values = data_array["lat"].values
    longitude_values = data_array["lon"].values
    latitude_step = float(numpy.abs(latitude_values[1] - latitude_values[0]))
    longitude_step = float(numpy.abs(longitude_values[1] - longitude_values[0]))
    latitude_target = numpy.arange(latitude_values.min(), latitude_values.max() + latitude_step, latitude_step)
    longitude_target = numpy.arange(longitude_values.min(), longitude_values.max() + longitude_step, longitude_step)
    return data_array.interp(lat=latitude_target, lon=longitude_target, method="linear")


def _latlon_deg_to_meters(data_array: xarray.DataArray) -> xarray.DataArray:
    longitude_deltas, latitude_deltas = lat_lon_grid_deltas(
        data_array["lon"].values * units.degree,
        data_array["lat"].values * units.degree,
    )
    mean_longitude_delta_meters = float(numpy.mean(longitude_deltas.to("meter").magnitude))
    mean_latitude_delta_meters = float(numpy.mean(latitude_deltas.to("meter").magnitude))
    longitude_values_meters = numpy.arange(data_array.sizes["lon"], dtype=numpy.float64) * mean_longitude_delta_meters
    latitude_values_meters = numpy.arange(data_array.sizes["lat"], dtype=numpy.float64) * mean_latitude_delta_meters
    return data_array.assign_coords(lon=longitude_values_meters, lat=latitude_values_meters)


def _time_to_days(data_array: xarray.DataArray) -> xarray.DataArray:
    if "time" not in data_array.coords:
        time_values = numpy.arange(data_array.sizes["time"], dtype=numpy.float32)
        return data_array.assign_coords(time=time_values)
    time_values = data_array["time"].values
    if numpy.issubdtype(time_values.dtype, numpy.datetime64):
        base_time = time_values.min()
        time_values = ((time_values - base_time) / numpy.timedelta64(1, "D")).astype(numpy.float32)
    else:
        time_values = numpy.asarray(time_values, dtype=numpy.float32)
    return data_array.assign_coords(time=time_values)


def _prepare_coordinate_psd_dataarray(
    dataset: xarray.Dataset,
    variable: Variable | str,
    first_day_index: int = 0,
    depth_selector: int | float | None = None,
) -> xarray.DataArray:
    variable_key = _as_variable_key(variable)
    data_array = rename_dataset_with_standard_names(dataset)[variable_key]
    data_array = _select_first_day_and_depth(
        data_array=data_array,
        first_day_index=first_day_index,
        depth_selector=depth_selector,
    )
    data_array = _rename_psd_dimensions(data_array)
    if "time" not in data_array.dims:
        data_array = data_array.expand_dims(time=[0.0])
    data_array = _wrap_longitudes(data_array)
    data_array = data_array.sortby("lat").sortby("time")
    data_array = _regularize_rectilinear_grid(data_array)
    return data_array


def _finalize_psd_dataarray(data_array: xarray.DataArray) -> xarray.DataArray:
    data_array = _latlon_deg_to_meters(data_array)
    data_array = _time_to_days(data_array)
    return data_array.astype(numpy.float32)


def prepare_psd_dataarray(
    dataset: xarray.Dataset,
    variable: Variable | str,
    first_day_index: int = 0,
    depth_selector: int | float | None = None,
) -> xarray.DataArray:
    return _finalize_psd_dataarray(
        _prepare_coordinate_psd_dataarray(
            dataset=dataset,
            variable=variable,
            first_day_index=first_day_index,
            depth_selector=depth_selector,
        )
    )


def _fill_psd_nans(data_array: xarray.DataArray, fill_value: float | None) -> xarray.DataArray:
    if fill_value is None:
        return data_array
    return data_array.fillna(fill_value)


def _zonal_longitude_psd_from_prepared_dataarray(
    data_array: xarray.DataArray,
    fill_value: float | None = DEFAULT_FILL_VALUE,
) -> xarray.DataArray:
    data_array = _fill_psd_nans(data_array, fill_value=fill_value)
    chunked_data_array = data_array.chunk({"time": 1, "lon": data_array.sizes["lon"], "lat": 1})
    power_spectrum = xrft.power_spectrum(
        chunked_data_array,
        dim=["lon"],
        detrend="linear",
        window="tukey",
        nfactor=2,
        window_correction=True,
        true_amplitude=True,
        truncate=True,
    )
    power_spectrum = power_spectrum.mean(dim=["lat"], skipna=True)
    power_spectrum = power_spectrum.sel(freq_lon=power_spectrum["freq_lon"] > 0).compute()
    power_spectrum.name = data_array.name
    if "time" in power_spectrum.dims:
        power_spectrum = power_spectrum.rename({"time": Dimension.LEAD_DAY_INDEX.key()})
        power_spectrum[Dimension.LEAD_DAY_INDEX.key()].attrs["long_name"] = "Lead day"
    return power_spectrum


def _positive_sorted_power_spectrum(power_spectrum: xarray.DataArray) -> xarray.DataArray:
    return power_spectrum.sel(freq_lon=power_spectrum["freq_lon"] > 0).sortby("freq_lon")


def _wavelength_band_limits_km(power_spectrum: xarray.DataArray) -> tuple[float, float]:
    frequencies = numpy.asarray(power_spectrum["freq_lon"].values, dtype=float)
    positive_frequencies = frequencies[numpy.isfinite(frequencies) & (frequencies > 0)]
    wavelength_km = 1.0 / positive_frequencies / 1000.0
    return float(numpy.nanmin(wavelength_km)), float(numpy.nanmax(wavelength_km))


def default_zonal_wavelength_bands_km(
    power_spectrum: xarray.DataArray,
    regional_scale_wavelength_threshold_km: float = DEFAULT_REGIONAL_SCALE_WAVELENGTH_THRESHOLD_KM,
    large_scale_wavelength_threshold_km: float = DEFAULT_LARGE_SCALE_WAVELENGTH_THRESHOLD_KM,
) -> dict[str, tuple[float, float]]:
    minimum_wavelength_km, maximum_wavelength_km = _wavelength_band_limits_km(power_spectrum)
    candidate_bands = [
        (
            DEFAULT_LARGE_SCALE_BAND_NAME,
            max(minimum_wavelength_km, large_scale_wavelength_threshold_km),
            maximum_wavelength_km,
        ),
        (
            DEFAULT_REGIONAL_SCALE_BAND_NAME,
            max(minimum_wavelength_km, regional_scale_wavelength_threshold_km),
            min(maximum_wavelength_km, large_scale_wavelength_threshold_km),
        ),
        (
            DEFAULT_NEAR_GRID_SCALE_BAND_NAME,
            minimum_wavelength_km,
            min(maximum_wavelength_km, regional_scale_wavelength_threshold_km),
        ),
    ]
    return {
        band_name: (minimum_band_wavelength_km, maximum_band_wavelength_km)
        for band_name, minimum_band_wavelength_km, maximum_band_wavelength_km in candidate_bands
        if maximum_band_wavelength_km > minimum_band_wavelength_km
    }


def _frequency_interval_from_wavelength_band_km(
    wavelength_band_km: tuple[float, float],
) -> tuple[float, float]:
    minimum_wavelength_km, maximum_wavelength_km = sorted(wavelength_band_km)
    minimum_frequency = 1.0 / (maximum_wavelength_km * 1000.0)
    maximum_frequency = 1.0 / (minimum_wavelength_km * 1000.0)
    return minimum_frequency, maximum_frequency


def zonal_longitude_band_energy_from_spectrum(
    power_spectrum: xarray.DataArray,
    wavelength_band_km: tuple[float, float],
) -> xarray.DataArray:
    positive_power_spectrum = _positive_sorted_power_spectrum(power_spectrum)
    minimum_frequency, maximum_frequency = _frequency_interval_from_wavelength_band_km(wavelength_band_km)
    band_power_spectrum = positive_power_spectrum.sel(freq_lon=slice(minimum_frequency, maximum_frequency))
    if band_power_spectrum.sizes.get("freq_lon", 0) == 0:
        if Dimension.LEAD_DAY_INDEX.key() in positive_power_spectrum.dims:
            lead_day_coordinate = positive_power_spectrum[Dimension.LEAD_DAY_INDEX.key()]
            nan_values = numpy.full(positive_power_spectrum.sizes[Dimension.LEAD_DAY_INDEX.key()], numpy.nan)
            return xarray.DataArray(
                nan_values,
                dims=[Dimension.LEAD_DAY_INDEX.key()],
                coords={Dimension.LEAD_DAY_INDEX.key(): lead_day_coordinate},
            )
        return xarray.DataArray(numpy.nan)
    return band_power_spectrum.integrate("freq_lon")


def zonal_longitude_band_energy_fraction_from_spectrum(
    power_spectrum: xarray.DataArray,
    wavelength_band_km: tuple[float, float],
) -> xarray.DataArray:
    positive_power_spectrum = _positive_sorted_power_spectrum(power_spectrum)
    total_energy = positive_power_spectrum.integrate("freq_lon")
    band_energy = zonal_longitude_band_energy_from_spectrum(power_spectrum, wavelength_band_km)
    return band_energy / total_energy.where(total_energy > 0)


def zonal_longitude_psd_metrics_from_spectrum(
    power_spectrum: xarray.DataArray,
    wavelength_bands_km: dict[str, tuple[float, float]] | None = None,
    numerator_band_name: str = DEFAULT_NEAR_GRID_SCALE_BAND_NAME,
    denominator_band_name: str = DEFAULT_REGIONAL_SCALE_BAND_NAME,
) -> pandas.DataFrame:
    resolved_wavelength_bands_km = wavelength_bands_km or default_zonal_wavelength_bands_km(power_spectrum)
    band_energies = {
        band_name: zonal_longitude_band_energy_from_spectrum(power_spectrum, wavelength_band_km)
        for band_name, wavelength_band_km in resolved_wavelength_bands_km.items()
    }
    band_energy_fractions = {
        band_name: zonal_longitude_band_energy_fraction_from_spectrum(power_spectrum, wavelength_band_km)
        for band_name, wavelength_band_km in resolved_wavelength_bands_km.items()
    }

    lead_day_count = (
        power_spectrum.sizes[Dimension.LEAD_DAY_INDEX.key()]
        if Dimension.LEAD_DAY_INDEX.key() in power_spectrum.dims
        else 1
    )
    metrics_by_row_label: dict[str, numpy.ndarray] = {}
    for band_name, wavelength_band_km in resolved_wavelength_bands_km.items():
        minimum_wavelength_km, maximum_wavelength_km = wavelength_band_km
        metrics_by_row_label[
            f"{band_name} band-integrated energy ({minimum_wavelength_km:.0f}-{maximum_wavelength_km:.0f} km)"
        ] = numpy.asarray(band_energies[band_name].values, dtype=float)
        metrics_by_row_label[
            f"{band_name} band energy fraction ({minimum_wavelength_km:.0f}-{maximum_wavelength_km:.0f} km)"
        ] = numpy.asarray(band_energy_fractions[band_name].values, dtype=float)

    if numerator_band_name in band_energies and denominator_band_name in band_energies:
        denominator = band_energies[denominator_band_name]
        small_scale_noise_index = band_energies[numerator_band_name] / denominator.where(denominator > 0)
        metrics_by_row_label[
            f"{numerator_band_name} noise index ({numerator_band_name} / {denominator_band_name})"
        ] = numpy.asarray(small_scale_noise_index.values, dtype=float)

    return pandas.DataFrame(
        metrics_by_row_label,
        index=lead_day_labels(1, lead_day_count),
    ).T


def zonal_longitude_psd_metrics(
    dataset: xarray.Dataset,
    variable: Variable | str,
    first_day_index: int = 0,
    depth_selector: int | float | None = None,
    fill_value: float | None = DEFAULT_FILL_VALUE,
    wavelength_bands_km: dict[str, tuple[float, float]] | None = None,
    numerator_band_name: str = DEFAULT_NEAR_GRID_SCALE_BAND_NAME,
    denominator_band_name: str = DEFAULT_REGIONAL_SCALE_BAND_NAME,
) -> pandas.DataFrame:
    power_spectrum = zonal_longitude_psd(
        dataset=dataset,
        variable=variable,
        first_day_index=first_day_index,
        depth_selector=depth_selector,
        fill_value=fill_value,
    )
    return zonal_longitude_psd_metrics_from_spectrum(
        power_spectrum=power_spectrum,
        wavelength_bands_km=wavelength_bands_km,
        numerator_band_name=numerator_band_name,
        denominator_band_name=denominator_band_name,
    )


def zonal_longitude_psd(
    dataset: xarray.Dataset,
    variable: Variable | str,
    first_day_index: int = 0,
    depth_selector: int | float | None = None,
    fill_value: float | None = DEFAULT_FILL_VALUE,
) -> xarray.DataArray:
    prepared_data_array = prepare_psd_dataarray(
        dataset=dataset,
        variable=variable,
        first_day_index=first_day_index,
        depth_selector=depth_selector,
    )
    return _zonal_longitude_psd_from_prepared_dataarray(prepared_data_array, fill_value=fill_value)


def zonal_longitude_psd_pair(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variable: Variable | str,
    first_day_index: int = 0,
    depth_selector: int | float | None = None,
    fill_value: float | None = DEFAULT_FILL_VALUE,
) -> tuple[xarray.DataArray, xarray.DataArray]:
    challenger_data_array = _prepare_coordinate_psd_dataarray(
        dataset=challenger_dataset,
        variable=variable,
        first_day_index=first_day_index,
        depth_selector=depth_selector,
    )
    reference_data_array = _prepare_coordinate_psd_dataarray(
        dataset=reference_dataset,
        variable=variable,
        first_day_index=first_day_index,
        depth_selector=depth_selector,
    )
    challenger_data_array, reference_data_array = xarray.align(
        challenger_data_array,
        reference_data_array,
        join="inner",
    )
    challenger_data_array = _finalize_psd_dataarray(challenger_data_array)
    reference_data_array = _finalize_psd_dataarray(reference_data_array)
    return (
        _zonal_longitude_psd_from_prepared_dataarray(challenger_data_array, fill_value=fill_value),
        _zonal_longitude_psd_from_prepared_dataarray(reference_data_array, fill_value=fill_value),
    )
