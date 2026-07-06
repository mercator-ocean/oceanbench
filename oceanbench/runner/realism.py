# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""The realism battery: spectra, activity ratio and eddy-census metrics.

This module turns the ported diagnostics (``oceanbench.core.psd`` and
``oceanbench.core.eddies``) into long-format ``scores.parquet`` records
(contracts.md §3.2) and the structured data the insight-artifact writers
(``oceanbench.publish.insights``) serialize into the ``spectra`` and ``eddies``
payloads (contracts.md §4).

Emission granularity. Spectra and eddy metrics are aggregate over the forecast
starts *by nature* (contracts.md §1: batch precomputes only aggregates over the
52-start ensemble and the non-browser eddy algorithm). Power spectra are computed
per start and averaged; eddy metrics are averaged over the processed starts. Every
realism record therefore carries ``start_date = None``. The eddy insight *census*
(polygons cannot be averaged) is emitted for a single representative start.

Spectral-band naming. Branch 249's zonal PSD bands are ``Large scale`` (≥ 2000 km),
``Regional scale`` (500–2000 km) and ``Near-grid scale`` (< 500 km). The contract
(§3.2) names the band column ``large`` / ``regional`` / ``mesoscale``; the map below
is the single source of truth. The contract's ``mesoscale`` is 249's sub-500 km
``Near-grid scale`` band — the scale at which eddy-resolving models carry mesoscale
energy. At 1° the grid cannot resolve < 500 km, so the ``mesoscale`` band is narrow
and degenerate (see the Phase-4 smoke).
"""

from dataclasses import dataclass, field

import numpy
import pandas
import xarray

from oceanbench.core import eddies as eddies_core
from oceanbench.core.version import __version__ as OCEANBENCH_VERSION
from oceanbench.core import psd as psd_core
from oceanbench.core.dataset_utils import VARIABLE_METADATA, Dimension, Variable
from oceanbench.runner import records

SEA_SURFACE_HEIGHT_KEY = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()

BAND_LARGE = "large"
BAND_REGIONAL = "regional"
BAND_MESOSCALE = "mesoscale"

_BAND_NAME_FROM_ZONAL_BAND = {
    psd_core.DEFAULT_LARGE_SCALE_BAND_NAME: BAND_LARGE,
    psd_core.DEFAULT_REGIONAL_SCALE_BAND_NAME: BAND_REGIONAL,
    psd_core.DEFAULT_NEAR_GRID_SCALE_BAND_NAME: BAND_MESOSCALE,
}

_EFFECTIVE_RESOLUTION_POWER_RATIO = 0.5
_MAXIMUM_CONTOUR_POINT_COUNT = 64
_COORDINATE_ROUNDING_DECIMALS = 4


@dataclass(frozen=True)
class RealismResult:
    records: list[dict]
    spectra_entries: list[dict]
    eddy_census: list[dict]
    flags: list[str] = field(default_factory=list)


def _as_variable_key(variable: Variable | str) -> str:
    return variable.key() if isinstance(variable, Variable) else variable


def _resolved_start_indices(dataset: xarray.Dataset, start_indices: list[int] | None) -> list[int]:
    start_count = dataset.sizes.get(Dimension.FIRST_DAY_DATETIME.key(), 1)
    if start_indices is None:
        return list(range(start_count))
    return [index for index in start_indices if 0 <= index < start_count]


def _available_lead_days(dataset: xarray.Dataset, lead_days: tuple[int, ...]) -> list[int]:
    lead_count = dataset.sizes.get(Dimension.LEAD_DAY_INDEX.key(), 1)
    return [lead_day for lead_day in lead_days if 1 <= lead_day <= lead_count]


def _mean_spectrum_over_starts(
    dataset: xarray.Dataset,
    variable: Variable | str,
    start_indices: list[int],
) -> xarray.DataArray:
    per_start_spectra = [
        psd_core.zonal_longitude_psd(dataset, variable, first_day_index=start_index) for start_index in start_indices
    ]
    aligned_spectra = xarray.align(*per_start_spectra, join="inner")
    return xarray.concat(aligned_spectra, dim="_start").mean(dim="_start")


def _contract_wavelength_bands(spectrum: xarray.DataArray) -> dict[str, tuple[float, float]]:
    zonal_bands = psd_core.default_zonal_wavelength_bands_km(spectrum)
    return {
        _BAND_NAME_FROM_ZONAL_BAND[zonal_band_name]: band_limits for zonal_band_name, band_limits in zonal_bands.items()
    }


def _spectrum_value_at_lead_day(spectrum_over_leads: xarray.DataArray, lead_day: int) -> float:
    lead_day_key = Dimension.LEAD_DAY_INDEX.key()
    if lead_day_key in spectrum_over_leads.dims:
        return float(spectrum_over_leads.isel({lead_day_key: lead_day - 1}).values)
    return float(spectrum_over_leads.values)


def _wavelength_kilometres(spectrum: xarray.DataArray) -> numpy.ndarray:
    frequencies = numpy.asarray(spectrum["freq_lon"].values, dtype=float)
    return 1.0 / frequencies / 1000.0


def _effective_resolution_kilometres(
    challenger_spectrum: xarray.DataArray,
    reference_spectrum: xarray.DataArray,
    lead_day: int,
) -> float | None:
    lead_day_key = Dimension.LEAD_DAY_INDEX.key()
    challenger_curve = challenger_spectrum.isel({lead_day_key: lead_day - 1})
    reference_curve = reference_spectrum.isel({lead_day_key: lead_day - 1})
    frequencies = numpy.asarray(challenger_curve["freq_lon"].values, dtype=float)
    ascending_order = numpy.argsort(frequencies)
    frequencies = frequencies[ascending_order]
    challenger_power = numpy.asarray(challenger_curve.values, dtype=float)[ascending_order]
    reference_power = numpy.asarray(reference_curve.values, dtype=float)[ascending_order]
    wavelength_km = 1.0 / frequencies / 1000.0
    ratio = numpy.where(reference_power > 0, challenger_power / reference_power, numpy.nan)
    return _first_downward_ratio_crossing_wavelength(wavelength_km, ratio)


def _first_downward_ratio_crossing_wavelength(
    wavelength_km: numpy.ndarray,
    ratio: numpy.ndarray,
) -> float | None:
    # Walking from the largest scales (lowest frequency) to the smallest, find the
    # first wavelength where the challenger/reference power ratio drops through 0.5.
    finite_mask = numpy.isfinite(ratio)
    for position in range(len(ratio) - 1):
        if not (finite_mask[position] and finite_mask[position + 1]):
            continue
        upper_ratio = ratio[position]
        lower_ratio = ratio[position + 1]
        if upper_ratio >= _EFFECTIVE_RESOLUTION_POWER_RATIO > lower_ratio:
            ratio_span = upper_ratio - lower_ratio
            if ratio_span == 0:
                return float(wavelength_km[position])
            interpolation_weight = (upper_ratio - _EFFECTIVE_RESOLUTION_POWER_RATIO) / ratio_span
            crossing_wavelength = wavelength_km[position] + interpolation_weight * (
                wavelength_km[position + 1] - wavelength_km[position]
            )
            return float(crossing_wavelength)
    return None


def _area_weighted_standard_deviation(anomaly: xarray.DataArray, latitude_key: str) -> float:
    latitude_radians = numpy.deg2rad(anomaly[latitude_key])
    latitude_weights = numpy.cos(latitude_radians)
    broadcast_weights = latitude_weights.broadcast_like(anomaly).where(numpy.isfinite(anomaly))
    finite_anomaly = anomaly.where(numpy.isfinite(anomaly))
    weight_total = broadcast_weights.sum()
    weighted_mean = (finite_anomaly * broadcast_weights).sum() / weight_total
    weighted_variance = (((finite_anomaly - weighted_mean) ** 2) * broadcast_weights).sum() / weight_total
    return float(numpy.sqrt(float(weighted_variance.values)))


def _activity_ratio_for_lead_day(
    challenger_field: xarray.DataArray,
    reference_field: xarray.DataArray,
    latitude_key: str,
) -> float | None:
    start_key = Dimension.FIRST_DAY_DATETIME.key()
    challenger_anomaly = challenger_field - challenger_field.mean(dim=start_key)
    reference_anomaly = reference_field - reference_field.mean(dim=start_key)
    challenger_std = _area_weighted_standard_deviation(challenger_anomaly, latitude_key)
    reference_std = _area_weighted_standard_deviation(reference_anomaly, latitude_key)
    if not numpy.isfinite(reference_std) or reference_std == 0:
        return None
    return challenger_std / reference_std


def _spectra_metric_records_for_reference(
    challenger_spectrum: xarray.DataArray,
    reference_spectrum: xarray.DataArray,
    error_spectrum: xarray.DataArray,
    *,
    reference_name: str,
    variable_key: str,
    unit: str,
    lead_days: list[int],
    context: records.RunContext,
    region: str,
) -> tuple[list[dict], list[dict]]:
    error_bands = _contract_wavelength_bands(error_spectrum)
    metric_records: list[dict] = []
    spectra_entries: list[dict] = []
    for lead_day in lead_days:
        effective_resolution = _effective_resolution_kilometres(challenger_spectrum, reference_spectrum, lead_day)
        metric_records.append(
            records.realism_record(
                context=context,
                metric=records.METRIC_EFFECTIVE_RESOLUTION_KILOMETRES,
                value=effective_resolution,
                unit="km",
                reference=reference_name,
                variable=variable_key,
                lead_day=lead_day,
            )
        )
        for band_name, band_limits in error_bands.items():
            band_energy = psd_core.zonal_longitude_band_energy_from_spectrum(error_spectrum, band_limits)
            metric_records.append(
                records.realism_record(
                    context=context,
                    metric=records.METRIC_ERROR_SPECTRUM_BAND_ENERGY,
                    value=_spectrum_value_at_lead_day(band_energy, lead_day),
                    unit=f"{unit}^2",
                    reference=reference_name,
                    variable=variable_key,
                    lead_day=lead_day,
                    band=band_name,
                )
            )
        spectra_entries.append(
            _spectrum_entry(
                challenger_spectrum,
                reference_spectrum,
                error_spectrum,
                reference_name=reference_name,
                variable_key=variable_key,
                unit=unit,
                lead_day=lead_day,
                region=region,
            )
        )
    return metric_records, spectra_entries


def _spectrum_entry(
    challenger_spectrum: xarray.DataArray,
    reference_spectrum: xarray.DataArray,
    error_spectrum: xarray.DataArray,
    *,
    reference_name: str,
    variable_key: str,
    unit: str,
    lead_day: int,
    region: str,
) -> dict:
    lead_day_key = Dimension.LEAD_DAY_INDEX.key()
    challenger_curve = challenger_spectrum.isel({lead_day_key: lead_day - 1})
    reference_curve = reference_spectrum.isel({lead_day_key: lead_day - 1})
    error_curve = error_spectrum.isel({lead_day_key: lead_day - 1})
    wavelength_km = _wavelength_kilometres(challenger_curve)
    return {
        "variable": variable_key,
        "region": region,
        "lead_day": lead_day,
        "reference": reference_name,
        "unit": f"{unit}^2",
        "wavelength": _nullable_float_list(wavelength_km),
        "challenger_power": _nullable_float_list(challenger_curve.values),
        "reference_power": _nullable_float_list(reference_curve.values),
        "error_power": _nullable_float_list(error_curve.values),
    }


def _nullable_float_list(values: numpy.ndarray) -> list[float | None]:
    return [None if not numpy.isfinite(value) else float(value) for value in numpy.asarray(values, dtype=float)]


def _band_fraction_records(
    challenger_spectrum: xarray.DataArray,
    *,
    variable_key: str,
    lead_days: list[int],
    context: records.RunContext,
) -> list[dict]:
    wavelength_bands = _contract_wavelength_bands(challenger_spectrum)
    band_fractions = {
        band_name: psd_core.zonal_longitude_band_energy_fraction_from_spectrum(challenger_spectrum, band_limits)
        for band_name, band_limits in wavelength_bands.items()
    }
    return [
        records.realism_record(
            context=context,
            metric=records.METRIC_PSD_BAND_ENERGY_FRACTION,
            value=_spectrum_value_at_lead_day(band_fraction, lead_day),
            unit="1",
            variable=variable_key,
            lead_day=lead_day,
            band=band_name,
        )
        for band_name, band_fraction in band_fractions.items()
        for lead_day in lead_days
    ]


def _aligned_surface_fields(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variable_key: str,
) -> tuple[xarray.DataArray, xarray.DataArray]:
    challenger_field = _surface_field(challenger_dataset, variable_key)
    reference_field = _surface_field(reference_dataset, variable_key)
    return xarray.align(challenger_field, reference_field, join="inner")


def _surface_field(dataset: xarray.Dataset, variable_key: str) -> xarray.DataArray:
    field = dataset[variable_key]
    if Dimension.DEPTH.key() in field.dims:
        field = field.isel({Dimension.DEPTH.key(): 0})
    return field


def _difference_dataset(
    challenger_field: xarray.DataArray,
    reference_field: xarray.DataArray,
    variable_key: str,
) -> xarray.Dataset:
    difference = challenger_field - reference_field
    difference_dataset = difference.to_dataset(name=variable_key)
    difference_dataset[variable_key].attrs["standard_name"] = variable_key
    return difference_dataset


def _spectra_and_activity_records(
    challenger_dataset: xarray.Dataset,
    reference_datasets: dict[str, xarray.Dataset],
    *,
    variable: Variable | str,
    region: str,
    context: records.RunContext,
    lead_days: list[int],
    start_indices: list[int],
) -> tuple[list[dict], list[dict], list[str]]:
    variable_key = _as_variable_key(variable)
    unit = VARIABLE_METADATA[variable_key][1]
    flags: list[str] = []

    own_challenger_spectrum = _mean_spectrum_over_starts(challenger_dataset, variable, start_indices)
    metric_records = _band_fraction_records(
        own_challenger_spectrum,
        variable_key=variable_key,
        lead_days=lead_days,
        context=context,
    )
    spectra_entries: list[dict] = []

    for reference_name, reference_dataset in reference_datasets.items():
        challenger_field, reference_field = _aligned_surface_fields(challenger_dataset, reference_dataset, variable_key)
        challenger_surface_dataset = challenger_field.to_dataset(name=variable_key)
        reference_surface_dataset = reference_field.to_dataset(name=variable_key)
        difference_dataset = _difference_dataset(challenger_field, reference_field, variable_key)

        challenger_spectrum = _mean_spectrum_over_starts(challenger_surface_dataset, variable, start_indices)
        reference_spectrum = _mean_spectrum_over_starts(reference_surface_dataset, variable, start_indices)
        error_spectrum = _mean_spectrum_over_starts(difference_dataset, variable, start_indices)

        reference_metric_records, reference_spectra_entries = _spectra_metric_records_for_reference(
            challenger_spectrum,
            reference_spectrum,
            error_spectrum,
            reference_name=reference_name,
            variable_key=variable_key,
            unit=unit,
            lead_days=lead_days,
            context=context,
            region=region,
        )
        metric_records.extend(reference_metric_records)
        spectra_entries.extend(reference_spectra_entries)

        latitude_key = challenger_field[Dimension.LATITUDE.key()].name
        for lead_day in lead_days:
            activity_ratio = _activity_ratio_for_lead_day(
                challenger_field.isel({Dimension.LEAD_DAY_INDEX.key(): lead_day - 1}),
                reference_field.isel({Dimension.LEAD_DAY_INDEX.key(): lead_day - 1}),
                latitude_key,
            )
            metric_records.append(
                records.realism_record(
                    context=context,
                    metric=records.METRIC_ACTIVITY_RATIO,
                    value=activity_ratio,
                    unit="1",
                    reference=reference_name,
                    variable=variable_key,
                    lead_day=lead_day,
                )
            )

    if BAND_MESOSCALE not in _contract_wavelength_bands(own_challenger_spectrum):
        flags.append(
            "psd mesoscale band absent: grid cannot resolve < 500 km wavelengths "
            "(expected at 1 degree; full mesoscale validation needs 1/12 degree data)."
        )
    return metric_records, spectra_entries, flags


def _eddy_records_and_census(
    challenger_dataset: xarray.Dataset,
    reference_datasets: dict[str, xarray.Dataset],
    *,
    context: records.RunContext,
    lead_days: list[int],
    eddy_start_indices: list[int],
    apply_contour_filtering: bool,
) -> tuple[list[dict], list[dict], list[str]]:
    lead_day_indices = [lead_day - 1 for lead_day in lead_days]
    metric_records: list[dict] = []
    census: list[dict] = []
    flags: list[str] = []

    challenger_count_emitted = False
    for reference_name, reference_dataset in reference_datasets.items():
        aggregated = _aggregate_eddy_statistics_over_starts(
            challenger_dataset,
            reference_dataset,
            lead_day_indices=lead_day_indices,
            start_indices=eddy_start_indices,
            apply_contour_filtering=apply_contour_filtering,
        )
        for polarity in eddies_core.POLARITY_ORDER:
            for lead_day in lead_days:
                statistics = aggregated[(lead_day, polarity)]
                if not challenger_count_emitted:
                    metric_records.append(
                        records.realism_record(
                            context=context,
                            metric=records.METRIC_EDDY_COUNT,
                            value=statistics["challenger_count"],
                            unit="count",
                            variable=SEA_SURFACE_HEIGHT_KEY,
                            lead_day=lead_day,
                            polarity=polarity,
                        )
                    )
                hit_rate, miss_rate = _hit_and_miss_rate(statistics)
                metric_records.append(
                    records.realism_record(
                        context=context,
                        metric=records.METRIC_EDDY_HIT_RATE,
                        value=hit_rate,
                        unit="1",
                        reference=reference_name,
                        variable=SEA_SURFACE_HEIGHT_KEY,
                        lead_day=lead_day,
                        polarity=polarity,
                    )
                )
                metric_records.append(
                    records.realism_record(
                        context=context,
                        metric=records.METRIC_EDDY_MISS_RATE,
                        value=miss_rate,
                        unit="1",
                        reference=reference_name,
                        variable=SEA_SURFACE_HEIGHT_KEY,
                        lead_day=lead_day,
                        polarity=polarity,
                    )
                )
                metric_records.append(
                    records.realism_record(
                        context=context,
                        metric=records.METRIC_EDDY_MEAN_DISPLACEMENT_KILOMETRES,
                        value=statistics["mean_displacement_km"],
                        unit="km",
                        reference=reference_name,
                        variable=SEA_SURFACE_HEIGHT_KEY,
                        lead_day=lead_day,
                        polarity=polarity,
                    )
                )
        challenger_count_emitted = True
        census.append(
            _eddy_reference_census(
                challenger_dataset,
                reference_dataset,
                reference_name=reference_name,
                lead_days=lead_days,
                census_start_index=eddy_start_indices[0],
                apply_contour_filtering=apply_contour_filtering,
            )
        )

    total_detections = sum(statistics["challenger_count"] for statistics in aggregated.values())
    if total_detections == 0:
        flags.append(
            "eddy detection found zero challenger eddies (valid at 1 degree; the coarse "
            "grid rarely resolves closed mesoscale contours — needs 1/12 degree data)."
        )
    return metric_records, census, flags


def _aggregate_eddy_statistics_over_starts(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    *,
    lead_day_indices: list[int],
    start_indices: list[int],
    apply_contour_filtering: bool,
) -> dict[tuple[int, str], dict[str, float]]:
    per_start_statistics: dict[tuple[int, str], list[dict[str, float]]] = {
        (lead_day_index + 1, polarity): []
        for lead_day_index in lead_day_indices
        for polarity in eddies_core.POLARITY_ORDER
    }
    for start_index in start_indices:
        challenger_detections = eddies_core.detect_mesoscale_eddies(
            challenger_dataset, first_day_index=start_index, lead_day_indices=lead_day_indices
        )
        reference_detections = eddies_core.detect_mesoscale_eddies(
            reference_dataset, first_day_index=start_index, lead_day_indices=lead_day_indices
        )
        if apply_contour_filtering:
            challenger_detections = _contour_filtered_detections(challenger_dataset, challenger_detections, start_index)
            reference_detections = _contour_filtered_detections(reference_dataset, reference_detections, start_index)
        matches = eddies_core.match_mesoscale_eddies(challenger_detections, reference_detections)
        for lead_day_index in lead_day_indices:
            for polarity in eddies_core.POLARITY_ORDER:
                per_start_statistics[(lead_day_index + 1, polarity)].append(
                    _single_start_eddy_statistics(
                        challenger_detections,
                        reference_detections,
                        matches,
                        lead_day_index=lead_day_index,
                        polarity=polarity,
                    )
                )
    return {key: _mean_eddy_statistics(values) for key, values in per_start_statistics.items()}


def _single_start_eddy_statistics(
    challenger_detections: pandas.DataFrame,
    reference_detections: pandas.DataFrame,
    matches: pandas.DataFrame,
    *,
    lead_day_index: int,
    polarity: str,
) -> dict[str, float]:
    challenger_count = _polarity_lead_count(challenger_detections, lead_day_index, polarity)
    reference_count = _polarity_lead_count(reference_detections, lead_day_index, polarity)
    lead_polarity_matches = (
        matches.loc[
            (matches[eddies_core.LEAD_DAY_COLUMN] == lead_day_index)
            & (matches[eddies_core.POLARITY_COLUMN] == polarity)
        ]
        if not matches.empty
        else matches
    )
    hit_count = int(lead_polarity_matches.shape[0])
    mean_displacement_km = (
        float(lead_polarity_matches[eddies_core.DISTANCE_COLUMN].mean()) if hit_count > 0 else numpy.nan
    )
    return {
        "challenger_count": float(challenger_count),
        "reference_count": float(reference_count),
        "hit_count": float(hit_count),
        "mean_displacement_km": mean_displacement_km,
    }


def _polarity_lead_count(detections: pandas.DataFrame, lead_day_index: int, polarity: str) -> int:
    if detections.empty:
        return 0
    return int(
        detections.loc[
            (detections[eddies_core.LEAD_DAY_COLUMN] == lead_day_index)
            & (detections[eddies_core.POLARITY_COLUMN] == polarity)
        ].shape[0]
    )


def _mean_eddy_statistics(per_start_statistics: list[dict[str, float]]) -> dict[str, float]:
    challenger_count = numpy.nanmean([statistics["challenger_count"] for statistics in per_start_statistics])
    reference_count = numpy.nanmean([statistics["reference_count"] for statistics in per_start_statistics])
    hit_count = numpy.nanmean([statistics["hit_count"] for statistics in per_start_statistics])
    displacements = [
        statistics["mean_displacement_km"]
        for statistics in per_start_statistics
        if numpy.isfinite(statistics["mean_displacement_km"])
    ]
    mean_displacement_km = float(numpy.mean(displacements)) if displacements else numpy.nan
    return {
        "challenger_count": float(challenger_count),
        "reference_count": float(reference_count),
        "hit_count": float(hit_count),
        "mean_displacement_km": mean_displacement_km,
    }


def _hit_and_miss_rate(statistics: dict[str, float]) -> tuple[float | None, float | None]:
    reference_count = statistics["reference_count"]
    if reference_count <= 0:
        return None, None
    hit_rate = statistics["hit_count"] / reference_count
    return hit_rate, 1.0 - hit_rate


def _eddy_reference_census(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    *,
    reference_name: str,
    lead_days: list[int],
    census_start_index: int,
    apply_contour_filtering: bool,
) -> dict:
    lead_day_indices = [lead_day - 1 for lead_day in lead_days]
    challenger_detections = eddies_core.detect_mesoscale_eddies(
        challenger_dataset, first_day_index=census_start_index, lead_day_indices=lead_day_indices
    )
    reference_detections = eddies_core.detect_mesoscale_eddies(
        reference_dataset, first_day_index=census_start_index, lead_day_indices=lead_day_indices
    )
    if apply_contour_filtering:
        challenger_detections = _contour_filtered_detections(
            challenger_dataset, challenger_detections, census_start_index
        )
        reference_detections = _contour_filtered_detections(reference_dataset, reference_detections, census_start_index)
    matches = eddies_core.match_mesoscale_eddies(challenger_detections, reference_detections)
    challenger_contours = _contours(challenger_dataset, challenger_detections, census_start_index)
    reference_contours = _contours(reference_dataset, reference_detections, census_start_index)
    frames = [
        _census_frame(
            lead_day,
            challenger_detections,
            reference_detections,
            matches,
            challenger_contours,
            reference_contours,
        )
        for lead_day in lead_days
    ]
    return {
        "reference": reference_name,
        "parameters": {
            **eddies_core.default_eddy_detection_parameters(),
            "apply_contour_filtering": apply_contour_filtering,
            "oceanbench_version": OCEANBENCH_VERSION,
        },
        "frames": frames,
    }


def _contour_filtered_detections(
    dataset: xarray.Dataset,
    detections: pandas.DataFrame,
    first_day_index: int,
) -> pandas.DataFrame:
    contours = _contours(dataset, detections, first_day_index)
    return eddies_core.filter_mesoscale_eddy_detections_by_contours(detections, contours)


def _contours(
    dataset: xarray.Dataset,
    detections: pandas.DataFrame,
    first_day_index: int,
) -> pandas.DataFrame:
    if detections.empty:
        return detections.iloc[0:0]
    return eddies_core.mesoscale_eddy_contours_from_detections(detections, dataset, first_day_index=first_day_index)


def _census_frame(
    lead_day: int,
    challenger_detections: pandas.DataFrame,
    reference_detections: pandas.DataFrame,
    matches: pandas.DataFrame,
    challenger_contours: pandas.DataFrame,
    reference_contours: pandas.DataFrame,
) -> dict:
    lead_day_index = lead_day - 1
    lead_matches = matches.loc[matches[eddies_core.LEAD_DAY_COLUMN] == lead_day_index] if not matches.empty else matches
    matched_challenger_indices = set(lead_matches["challenger_detection_index"]) if not lead_matches.empty else set()
    matched_reference_indices = set(lead_matches["reference_detection_index"]) if not lead_matches.empty else set()
    match_entries = [
        {
            "challenger": _eddy_dict(
                int(match_row.challenger_detection_index), challenger_detections, challenger_contours
            ),
            "reference": _eddy_dict(int(match_row.reference_detection_index), reference_detections, reference_contours),
            "displacement_km": round(float(match_row.distance_km), _COORDINATE_ROUNDING_DECIMALS),
        }
        for match_row in lead_matches.itertuples(index=False)
    ]
    spurious = [
        _eddy_dict(detection_index, challenger_detections, challenger_contours)
        for detection_index in _lead_detection_indices(challenger_detections, lead_day_index)
        if detection_index not in matched_challenger_indices
    ]
    missed = [
        _eddy_dict(detection_index, reference_detections, reference_contours)
        for detection_index in _lead_detection_indices(reference_detections, lead_day_index)
        if detection_index not in matched_reference_indices
    ]
    return {"lead_day": lead_day, "matches": match_entries, "spurious": spurious, "missed": missed}


def _lead_detection_indices(detections: pandas.DataFrame, lead_day_index: int) -> list[int]:
    if detections.empty:
        return []
    return [int(index) for index in detections.loc[detections[eddies_core.LEAD_DAY_COLUMN] == lead_day_index].index]


def _eddy_dict(
    detection_index: int,
    detections: pandas.DataFrame,
    contours: pandas.DataFrame,
) -> dict:
    detection_row = detections.loc[detection_index]
    contour_latitudes, contour_longitudes = _contour_polygon(detection_index, contours)
    return {
        "id": int(detection_index),
        "latitude": round(float(detection_row[eddies_core.LATITUDE_COLUMN]), _COORDINATE_ROUNDING_DECIMALS),
        "longitude": round(float(detection_row[eddies_core.LONGITUDE_COLUMN]), _COORDINATE_ROUNDING_DECIMALS),
        "polarity": str(detection_row[eddies_core.POLARITY_COLUMN]),
        "contour_latitude": contour_latitudes,
        "contour_longitude": contour_longitudes,
    }


def _contour_polygon(
    detection_index: int,
    contours: pandas.DataFrame,
) -> tuple[list[float], list[float]]:
    if contours.empty:
        return [], []
    matching_contours = contours.loc[contours["detection_index"] == detection_index]
    if matching_contours.empty:
        return [], []
    contour_row = matching_contours.iloc[0]
    latitudes = numpy.asarray(contour_row[eddies_core.CONTOUR_LATITUDES_COLUMN], dtype=float)
    longitudes = numpy.asarray(contour_row[eddies_core.CONTOUR_LONGITUDES_COLUMN], dtype=float)
    latitudes, longitudes = _decimated_contour(latitudes, longitudes)
    return (
        [round(float(value), _COORDINATE_ROUNDING_DECIMALS) for value in latitudes],
        [round(float(value), _COORDINATE_ROUNDING_DECIMALS) for value in longitudes],
    )


def _decimated_contour(
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    if latitudes.size <= _MAXIMUM_CONTOUR_POINT_COUNT:
        return latitudes, longitudes
    stride = int(numpy.ceil(latitudes.size / _MAXIMUM_CONTOUR_POINT_COUNT))
    return latitudes[::stride], longitudes[::stride]


def compute_realism_battery(
    challenger_dataset: xarray.Dataset,
    reference_datasets: dict[str, xarray.Dataset],
    *,
    region: str,
    context: records.RunContext,
    variable: Variable | str = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    lead_days: tuple[int, ...] = (1, 5, 10),
    start_indices: list[int] | None = None,
    eddy_start_indices: list[int] | None = None,
    apply_eddy_contour_filtering: bool = eddies_core.DEFAULT_APPLY_CONTOUR_FILTERING,
) -> RealismResult:
    """Compute the realism battery for one (challenger, region) over the requested references.

    Spectra (band fractions, effective resolution, error-spectrum band energy) and the
    activity ratio are aggregated over ``start_indices`` (all starts when ``None``). Eddy
    metrics are aggregated over ``eddy_start_indices`` (the first start when ``None``), whose
    first entry also supplies the single-start eddy census returned for the insight artifact.
    Contour filtering defaults on so that metrics, matching and census share the core
    summary's closed-contour definition (area bounds in km² and convexity >= 0.75); set
    ``apply_eddy_contour_filtering=False`` to recover the raw-peak census of the
    already-published ``glonet_1_degree`` artifact. The census ``parameters`` object stamps
    the filtering mode and OceanBench version so artifacts stay distinguishable.
    ``variable`` selects the spectral/activity field (eddy detection is always on SSH).
    """
    resolved_start_indices = _resolved_start_indices(challenger_dataset, start_indices)
    resolved_eddy_start_indices = _resolved_start_indices(
        challenger_dataset, eddy_start_indices if eddy_start_indices is not None else [0]
    )
    resolved_lead_days = _available_lead_days(challenger_dataset, lead_days)

    metric_records, spectra_entries, spectra_flags = _spectra_and_activity_records(
        challenger_dataset,
        reference_datasets,
        variable=variable,
        region=region,
        context=context,
        lead_days=resolved_lead_days,
        start_indices=resolved_start_indices,
    )
    eddy_records, eddy_census, eddy_flags = _eddy_records_and_census(
        challenger_dataset,
        reference_datasets,
        context=context,
        lead_days=resolved_lead_days,
        eddy_start_indices=resolved_eddy_start_indices,
        apply_contour_filtering=apply_eddy_contour_filtering,
    )
    return RealismResult(
        records=metric_records + eddy_records,
        spectra_entries=spectra_entries,
        eddy_census=eddy_census,
        flags=spectra_flags + eddy_flags,
    )
