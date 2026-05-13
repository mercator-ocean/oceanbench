# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

import oceanbench
from oceanbench.core.dataset_utils import Dimension, Variable


def _sinusoidal_dataset(offset: float = 0.0) -> xarray.Dataset:
    first_days = numpy.array(["2024-01-03"], dtype="datetime64[D]")
    lead_days = numpy.arange(3)
    latitudes = numpy.array([-2.0, 0.0, 2.0])
    longitudes = numpy.linspace(0.0, 330.0, 12)
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
