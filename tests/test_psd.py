# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

import oceanbench
from oceanbench.core.dataset_utils import Dimension, Variable


def _sinusoidal_dataset(
    offset: float = 0.0,
    latitudes: numpy.ndarray | None = None,
    longitudes: numpy.ndarray | None = None,
) -> xarray.Dataset:
    first_days = numpy.array(["2024-01-03"], dtype="datetime64[D]")
    lead_days = numpy.arange(3)
    latitudes = numpy.array([-2.0, 0.0, 2.0]) if latitudes is None else latitudes
    longitudes = numpy.linspace(0.0, 330.0, 12) if longitudes is None else longitudes
    phase = numpy.deg2rad(longitudes)
    values = numpy.empty((len(first_days), len(lead_days), len(latitudes), len(longitudes)), dtype=float)
    for lead_day_index in range(len(lead_days)):
        values[:, lead_day_index, :, :] = (
            numpy.sin(phase * 2.0 + lead_day_index * 0.2)[None, None, :]
            + 0.25 * numpy.cos(phase * 4.0)[None, None, :]
            + offset
        )
    return xarray.Dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                values,
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): first_days,
            Dimension.LEAD_DAY_INDEX.key(): lead_days,
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


def test_zonal_longitude_psd_pair_returns_positive_metric_frequency_spectrum() -> None:
    challenger_spectrum, reference_spectrum = oceanbench.psd.zonal_longitude_psd_pair(
        _sinusoidal_dataset(),
        _sinusoidal_dataset(offset=0.1),
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    )

    assert challenger_spectrum.dims == (Dimension.LEAD_DAY_INDEX.key(), "freq_lon")
    assert challenger_spectrum.sizes[Dimension.LEAD_DAY_INDEX.key()] == 3
    assert challenger_spectrum.sizes["freq_lon"] > 0
    assert numpy.all(challenger_spectrum["freq_lon"].values > 0)
    assert numpy.isfinite(challenger_spectrum.values).any()
    assert reference_spectrum.sizes == challenger_spectrum.sizes


def test_zonal_longitude_psd_pair_is_unchanged_by_a_float32_encoding_difference() -> None:
    """A reference stored in float32 must yield the same spectrum, not a resampled one (issue #305).

    Before the grids were snapped, each side was regularized against its own coordinates and
    the inner join then intersected the two results, silently changing the sampling and with
    it the wavenumber axis. A 1/12-degree grid is not exactly representable in float32, so
    the round-trip below reproduces the encoding difference seen between CMEMS products.
    """
    twelfth_degree_longitudes = -180.0 + numpy.arange(48) / 12.0
    twelfth_degree_latitudes = -2.0 + numpy.arange(6) / 12.0
    challenger = _sinusoidal_dataset(latitudes=twelfth_degree_latitudes, longitudes=twelfth_degree_longitudes)
    reference = _sinusoidal_dataset(
        offset=0.1, latitudes=twelfth_degree_latitudes, longitudes=twelfth_degree_longitudes
    )
    float32_reference = reference.assign_coords(
        {
            Dimension.LONGITUDE.key(): twelfth_degree_longitudes.astype("float32").astype("float64"),
            Dimension.LATITUDE.key(): twelfth_degree_latitudes.astype("float32").astype("float64"),
        }
    )

    exact_spectra = oceanbench.psd.zonal_longitude_psd_pair(
        challenger, reference, Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID
    )
    float32_spectra = oceanbench.psd.zonal_longitude_psd_pair(
        challenger, float32_reference, Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID
    )

    for exact_spectrum, float32_spectrum in zip(exact_spectra, float32_spectra):
        assert float32_spectrum.sizes == exact_spectrum.sizes
        numpy.testing.assert_allclose(float32_spectrum["freq_lon"].values, exact_spectrum["freq_lon"].values, rtol=1e-6)
        numpy.testing.assert_allclose(float32_spectrum.values, exact_spectrum.values, rtol=1e-4)


def test_zonal_longitude_psd_metrics_exposes_wavelength_band_scores() -> None:
    power_spectrum = oceanbench.psd.zonal_longitude_psd(
        _sinusoidal_dataset(),
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    )
    wavelength_bands = oceanbench.psd.default_zonal_wavelength_bands_km(power_spectrum)

    metrics = oceanbench.psd.zonal_longitude_psd_metrics_from_spectrum(
        power_spectrum,
        wavelength_bands_km=wavelength_bands,
    )

    assert not metrics.empty
    assert all(column in metrics.columns for column in ["Lead day 1", "Lead day 2", "Lead day 3"])
    assert any("band-integrated energy" in row_label for row_label in metrics.index)


def test_prepare_psd_dataarray_keeps_float32_polar_grid_finite() -> None:
    first_days = numpy.array(["2024-01-03"], dtype="datetime64[D]")
    lead_days = numpy.array([0])
    latitudes = numpy.arange(2041, dtype=numpy.float32) * numpy.float32(1 / 12) - numpy.float32(80)
    longitudes = numpy.arange(4, dtype=numpy.float32) * numpy.float32(1 / 12)
    values = numpy.zeros((len(first_days), len(lead_days), len(latitudes), len(longitudes)), dtype=numpy.float32)
    dataset = xarray.Dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                values,
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): first_days,
            Dimension.LEAD_DAY_INDEX.key(): lead_days,
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )

    data_array = oceanbench.psd.prepare_psd_dataarray(dataset, Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID)

    assert numpy.isfinite(data_array["lat"].values).all()
    assert numpy.isfinite(data_array["lon"].values).all()
