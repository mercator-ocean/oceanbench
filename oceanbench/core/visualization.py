# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Mapping, Sequence
import math

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy
import xarray

from oceanbench.core import classIV
from oceanbench.core import class4_drifters
from oceanbench.core import eddies
from oceanbench.core import lagrangian_trajectory
from oceanbench.core import psd
from oceanbench.core import regions
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import DEPTH_BINS_BY_VARIABLE, DEPTH_BINS_DEFAULT, Dimension, Variable, VARIABLE_LABELS

DEFAULT_VARIABLES: tuple[Variable, ...] = (
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
    Variable.SEA_WATER_SALINITY,
    Variable.EASTWARD_SEA_WATER_VELOCITY,
    Variable.NORTHWARD_SEA_WATER_VELOCITY,
)
DEFAULT_PSD_VARIABLES: tuple[Variable, ...] = (
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
)

FIELD_CMAPS: dict[str, str] = {
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): "coolwarm",
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): "Spectral_r",
    Variable.SEA_WATER_SALINITY.key(): "viridis",
    Variable.MIXED_LAYER_DEPTH.key(): "jet",
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(): "RdBu_r",
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): "RdBu_r",
    Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY.key(): "RdBu_r",
    Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY.key(): "RdBu_r",
}

DEFAULT_CLASS4_DEPTH_BINS: dict[str, str] = {
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): "surface",
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): "SST",
    Variable.SEA_WATER_SALINITY.key(): "0-5m",
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(): "15m",
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): "15m",
}

DIVERGING_VARIABLES: set[str] = {
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
    Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY.key(),
    Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY.key(),
}

ERROR_CMAP = "jet"
ERROR_CLIP_QUANTILE = 0.9
DEFAULT_GALLERY_COLUMNS = 2
LONGITUDE_MIN = -180.0
LONGITUDE_MAX = 180.0
LONGITUDE_SPAN = LONGITUDE_MAX - LONGITUDE_MIN
DEFAULT_SCATTER_GRID_RESOLUTION_DEGREES = 2.0
DEFAULT_PSD_LEAD_DAY_INDICES: tuple[int, ...] = (0, 9)
DEFAULT_EDDY_LEAD_DAY_INDICES: tuple[int, ...] = (0, 9)
EDDY_REFERENCE_NAME = "GLORYS reanalysis"
EDDY_CYCLONE_COLOR = "#2563eb"
EDDY_ANTICYCLONE_COLOR = "#dc2626"


def _as_variable_key(variable: Variable | str) -> str:
    return variable.key() if isinstance(variable, Variable) else variable


def _as_variable_enum(variable: Variable | str) -> Variable:
    return variable if isinstance(variable, Variable) else next(item for item in Variable if item.key() == variable)


def _friendly_variable_label(variable_key: str) -> str:
    return VARIABLE_LABELS[variable_key].capitalize()


def _standard_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    return rename_dataset_with_standard_names(dataset)


def _select_depth(data_array: xarray.DataArray, depth_selector: int | float | None) -> xarray.DataArray:
    if Dimension.DEPTH.key() not in data_array.dims:
        return data_array
    if depth_selector is None:
        return data_array.isel({Dimension.DEPTH.key(): 0})
    if isinstance(depth_selector, int):
        return data_array.isel({Dimension.DEPTH.key(): depth_selector})
    return data_array.sel({Dimension.DEPTH.key(): depth_selector}, method="nearest")


def _select_map_field(
    dataset: xarray.Dataset,
    variable_key: str,
    first_day_index: int,
    lead_day_index: int,
    depth_selector: int | float | None,
) -> xarray.DataArray:
    standard_dataset = _standard_dataset(dataset)
    field = standard_dataset[variable_key]
    if Dimension.FIRST_DAY_DATETIME.key() in field.dims:
        field = field.isel({Dimension.FIRST_DAY_DATETIME.key(): first_day_index})
    if Dimension.LEAD_DAY_INDEX.key() in field.dims:
        field = field.isel({Dimension.LEAD_DAY_INDEX.key(): lead_day_index})
    field = _select_depth(field, depth_selector)
    return field.compute()


def _title_with_depth(variable_key: str, field: xarray.DataArray) -> str:
    label = _friendly_variable_label(variable_key)
    if Dimension.DEPTH.key() not in field.coords:
        return label
    depth_value = float(field[Dimension.DEPTH.key()].item())
    return f"{label} ({depth_value:.1f} m)"


def _min_max(arrays: Sequence[xarray.DataArray]) -> tuple[float, float]:
    minima = [float(numpy.nanmin(array.values)) for array in arrays]
    maxima = [float(numpy.nanmax(array.values)) for array in arrays]
    return min(minima), max(maxima)


def _finite_values(arrays: Sequence[xarray.DataArray]) -> numpy.ndarray:
    values = [numpy.asarray(array.values).ravel() for array in arrays]
    if not values:
        return numpy.array([])
    flattened = numpy.concatenate(values)
    return flattened[numpy.isfinite(flattened)]


def _field_norm(
    variable_key: str,
    arrays: Sequence[xarray.DataArray],
    cmap_override: str | None = None,
) -> tuple[str, Normalize | None]:
    cmap = cmap_override or FIELD_CMAPS.get(variable_key, "viridis")
    minimum, maximum = _min_max(arrays)
    if variable_key in DIVERGING_VARIABLES and minimum < 0 < maximum:
        return cmap, TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)
    return cmap, Normalize(vmin=minimum, vmax=maximum)


def _positive_error_norm(values: numpy.ndarray, quantile: float = ERROR_CLIP_QUANTILE) -> Normalize:
    finite_values = values[numpy.isfinite(values)]
    if finite_values.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmax = float(numpy.nanquantile(finite_values, quantile))
    if not numpy.isfinite(vmax) or vmax <= 0:
        vmax = float(numpy.nanmax(finite_values))
    if not numpy.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return Normalize(vmin=0.0, vmax=vmax)


def _wrap_longitudes(longitudes: numpy.ndarray) -> numpy.ndarray:
    wrapped_longitudes = ((longitudes - LONGITUDE_MIN) % LONGITUDE_SPAN) + LONGITUDE_MIN
    wrapped_longitudes = numpy.asarray(wrapped_longitudes, dtype=float)
    positive_dateline_mask = (wrapped_longitudes == LONGITUDE_MIN) & (numpy.asarray(longitudes) > 0)
    wrapped_longitudes[positive_dateline_mask] = LONGITUDE_MAX
    return wrapped_longitudes


def _domain_bounds(dataset: xarray.Dataset) -> tuple[float, float, float, float]:
    region = regions.region_from_dataset(dataset)
    if region is not None:
        return (
            float(region.minimum_longitude),
            float(region.maximum_longitude),
            float(region.minimum_latitude),
            float(region.maximum_latitude),
        )
    standard_dataset = _standard_dataset(dataset)
    longitude_values = _wrap_longitudes(standard_dataset[Dimension.LONGITUDE.key()].values)
    latitude_values = numpy.asarray(standard_dataset[Dimension.LATITUDE.key()].values, dtype=float)
    return (
        float(numpy.nanmin(longitude_values)),
        float(numpy.nanmax(longitude_values)),
        float(numpy.nanmin(latitude_values)),
        float(numpy.nanmax(latitude_values)),
    )


def _gallery_grid(item_count: int, max_columns: int = DEFAULT_GALLERY_COLUMNS) -> tuple[int, int]:
    column_count = min(max_columns, max(1, item_count))
    row_count = math.ceil(item_count / column_count)
    return row_count, column_count


def _resolved_psd_lead_day_positions(
    power_spectrum: xarray.DataArray,
    lead_day_indices: Sequence[int],
) -> list[int]:
    if Dimension.LEAD_DAY_INDEX.key() not in power_spectrum.dims:
        return []
    available_count = power_spectrum.sizes[Dimension.LEAD_DAY_INDEX.key()]
    return sorted({index for index in lead_day_indices if 0 <= index < available_count})


def _lead_day_label(power_spectrum: xarray.DataArray, lead_day_position: int) -> int:
    lead_day_value = power_spectrum[Dimension.LEAD_DAY_INDEX.key()].values[lead_day_position]
    return int(lead_day_value) + 1


def _wavelength_km_from_frequency(frequencies: numpy.ndarray) -> numpy.ndarray:
    return 1.0 / numpy.asarray(frequencies, dtype=float) / 1000.0


def _sorted_wavelength_slice(
    power_spectrum: xarray.DataArray,
    lead_day_position: int,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    lead_day_slice = power_spectrum.isel({Dimension.LEAD_DAY_INDEX.key(): lead_day_position})
    wavelength_km = _wavelength_km_from_frequency(lead_day_slice["freq_lon"].values)
    sort_indices = numpy.argsort(wavelength_km)
    wavelength_km = wavelength_km[sort_indices]
    power_values = numpy.asarray(lead_day_slice.values, dtype=float)[sort_indices]
    valid_mask = numpy.isfinite(wavelength_km) & numpy.isfinite(power_values) & (wavelength_km > 0) & (power_values > 0)
    return wavelength_km[valid_mask], power_values[valid_mask]


def _spatial_rmse_field(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variable_key: str,
    depth_selector: int | float | None,
) -> xarray.DataArray:
    challenger = _standard_dataset(challenger_dataset)[variable_key]
    reference = _standard_dataset(reference_dataset)[variable_key]
    challenger = _select_depth(challenger, depth_selector)
    reference = _select_depth(reference, depth_selector)
    challenger, reference = xarray.align(challenger, reference, join="inner")
    reduction_dimensions = [
        dimension
        for dimension in (Dimension.FIRST_DAY_DATETIME.key(), Dimension.LEAD_DAY_INDEX.key())
        if dimension in challenger.dims
    ]
    return numpy.sqrt(((challenger - reference) ** 2).mean(dim=reduction_dimensions)).compute()


def _plot_map(
    ax,
    field: xarray.DataArray,
    title: str,
    cmap: str,
    norm: Normalize | None = None,
):
    mesh = ax.pcolormesh(
        field[Dimension.LONGITUDE.key()].values,
        field[Dimension.LATITUDE.key()].values,
        field.values,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return mesh


def plot_surface_field_gallery(
    challenger_dataset: xarray.Dataset,
    variables: Sequence[Variable | str] = DEFAULT_VARIABLES,
    first_day_index: int = 0,
    lead_day_index: int = 0,
    depth_selectors: Mapping[str, int | float | None] | None = None,
    cmap_overrides: Mapping[str, str] | None = None,
    max_columns: int = DEFAULT_GALLERY_COLUMNS,
):
    depth_selectors = depth_selectors or {}
    cmap_overrides = cmap_overrides or {}
    resolved_variables = [_as_variable_key(variable) for variable in variables]
    row_count, column_count = _gallery_grid(len(resolved_variables), max_columns=max_columns)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(6.2 * column_count, 3.9 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    flattened_axes = axes.flatten()

    for axis in flattened_axes[len(resolved_variables) :]:
        axis.axis("off")

    for index, variable_key in enumerate(resolved_variables):
        depth_selector = depth_selectors.get(variable_key)
        field = _select_map_field(challenger_dataset, variable_key, first_day_index, lead_day_index, depth_selector)
        cmap, norm = _field_norm(variable_key, [field], cmap_override=cmap_overrides.get(variable_key))
        axis = flattened_axes[index]
        mesh = _plot_map(
            axis,
            field,
            title=_title_with_depth(variable_key, field),
            cmap=cmap,
            norm=norm,
        )
        figure.colorbar(mesh, ax=axis, shrink=0.9, pad=0.01)

    figure.suptitle(f"GLONET challenger fields for lead day {lead_day_index + 1}", fontsize=16)
    return figure


def plot_surface_field_comparison_gallery(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variable: Variable | str = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    reference_name: str = EDDY_REFERENCE_NAME,
    challenger_name: str = "GLONET",
    first_day_index: int = 0,
    lead_day_indices: Sequence[int] = DEFAULT_EDDY_LEAD_DAY_INDICES,
    depth_selector: int | float | None = None,
    cmap_override: str | None = None,
):
    variable_key = _as_variable_key(variable)
    resolved_lead_day_indices = sorted({lead_day_index for lead_day_index in lead_day_indices if lead_day_index >= 0})
    if not resolved_lead_day_indices:
        raise ValueError("lead_day_indices must contain at least one non-negative lead day index")

    field_pairs: list[tuple[int, xarray.DataArray, xarray.DataArray]] = []
    fields_for_norm: list[xarray.DataArray] = []
    for lead_day_index in resolved_lead_day_indices:
        challenger_field = _select_map_field(
            challenger_dataset,
            variable_key,
            first_day_index,
            lead_day_index,
            depth_selector,
        )
        reference_field = _select_map_field(
            reference_dataset,
            variable_key,
            first_day_index,
            lead_day_index,
            depth_selector,
        )
        field_pairs.append((lead_day_index, challenger_field, reference_field))
        fields_for_norm.extend([challenger_field, reference_field])

    cmap, norm = _field_norm(variable_key, fields_for_norm, cmap_override=cmap_override)
    figure, axes = plt.subplots(
        len(field_pairs),
        2,
        figsize=(16.0, 5.8 * len(field_pairs)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_index, (lead_day_index, challenger_field, reference_field) in enumerate(field_pairs):
        challenger_axis = axes[row_index, 0]
        challenger_mesh = _plot_map(
            challenger_axis,
            challenger_field,
            title=f"{challenger_name} lead day {lead_day_index + 1}",
            cmap=cmap,
            norm=norm,
        )

        reference_axis = axes[row_index, 1]
        reference_mesh = _plot_map(
            reference_axis,
            reference_field,
            title=f"{reference_name} lead day {lead_day_index + 1}",
            cmap=cmap,
            norm=norm,
        )

        figure.colorbar(
            reference_mesh if row_index else challenger_mesh,
            ax=[challenger_axis, reference_axis],
            shrink=0.9,
            pad=0.01,
            label=_friendly_variable_label(variable_key),
        )

    figure.suptitle(f"{_friendly_variable_label(variable_key)} fields", fontsize=16)
    return figure


def plot_spatial_rmse_gallery(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    reference_name: str,
    variables: Sequence[Variable | str] = DEFAULT_VARIABLES,
    depth_selectors: Mapping[str, int | float | None] | None = None,
    max_columns: int = DEFAULT_GALLERY_COLUMNS,
):
    depth_selectors = depth_selectors or {}
    resolved_variables = [_as_variable_key(variable) for variable in variables]
    row_count, column_count = _gallery_grid(len(resolved_variables), max_columns=max_columns)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(6.2 * column_count, 3.9 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    flattened_axes = axes.flatten()

    for axis in flattened_axes[len(resolved_variables) :]:
        axis.axis("off")

    for index, variable_key in enumerate(resolved_variables):
        depth_selector = depth_selectors.get(variable_key)
        field = _spatial_rmse_field(challenger_dataset, reference_dataset, variable_key, depth_selector)
        norm = _positive_error_norm(_finite_values([field]))
        axis = flattened_axes[index]
        mesh = _plot_map(
            axis,
            field,
            title=f"{_title_with_depth(variable_key, field)} RMSE vs {reference_name}",
            cmap=ERROR_CMAP,
            norm=norm,
        )
        figure.colorbar(mesh, ax=axis, shrink=0.9, pad=0.01)

    figure.suptitle("Spatial RMSE maps", fontsize=16)
    return figure


def plot_zonal_longitude_psd_comparison_gallery(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    reference_name: str,
    variables: Sequence[Variable | str] = DEFAULT_PSD_VARIABLES,
    lead_day_indices: Sequence[int] = DEFAULT_PSD_LEAD_DAY_INDICES,
    first_day_index: int = 0,
    depth_selectors: Mapping[str, int | float | None] | None = None,
    max_columns: int = DEFAULT_GALLERY_COLUMNS,
):
    depth_selectors = depth_selectors or {}
    resolved_variables = [_as_variable_key(variable) for variable in variables]
    row_count, column_count = _gallery_grid(len(resolved_variables), max_columns=max_columns)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(6.4 * column_count, 4.3 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    flattened_axes = axes.flatten()

    for axis in flattened_axes[len(resolved_variables) :]:
        axis.axis("off")

    line_styles = ["-", "--", ":", "-."]
    for index, variable_key in enumerate(resolved_variables):
        axis = flattened_axes[index]
        challenger_psd, reference_psd = psd.zonal_longitude_psd_pair(
            challenger_dataset=challenger_dataset,
            reference_dataset=reference_dataset,
            variable=variable_key,
            first_day_index=first_day_index,
            depth_selector=depth_selectors.get(variable_key),
        )
        resolved_lead_day_indices = _resolved_psd_lead_day_positions(challenger_psd, lead_day_indices)
        if not resolved_lead_day_indices:
            axis.set_axis_off()
            continue

        for style_index, lead_day_index in enumerate(resolved_lead_day_indices):
            line_style = line_styles[style_index % len(line_styles)]
            challenger_wavelength_km, challenger_values = _sorted_wavelength_slice(challenger_psd, lead_day_index)
            reference_wavelength_km, reference_values = _sorted_wavelength_slice(reference_psd, lead_day_index)
            axis.plot(
                challenger_wavelength_km,
                challenger_values,
                color="tab:blue",
                linestyle=line_style,
                linewidth=2.0,
                label=f"GLONET day {_lead_day_label(challenger_psd, lead_day_index)}",
            )
            axis.plot(
                reference_wavelength_km,
                reference_values,
                color="tab:orange",
                linestyle=line_style,
                linewidth=2.0,
                label=f"{reference_name} day {_lead_day_label(reference_psd, lead_day_index)}",
            )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.invert_xaxis()
        axis.grid(True, alpha=0.25, which="both")
        axis.set_xlabel("Zonal wavelength (km)")
        axis.set_ylabel("PSD")
        axis.set_title(_friendly_variable_label(variable_key))
        axis.legend()

    figure.suptitle(f"Global zonal PSD comparison against {reference_name}", fontsize=16)
    return figure


def _mesoscale_eddy_marker_style(polarity: str) -> tuple[str, str]:
    if polarity == eddies.CYCLONE:
        return EDDY_CYCLONE_COLOR, "v"
    return EDDY_ANTICYCLONE_COLOR, "^"


def _plot_mesoscale_eddy_contours(
    axis,
    contour_dataframe,
    linewidth: float = 0.9,
    alpha: float = 0.9,
) -> None:
    if len(contour_dataframe) == 0:
        return
    for _, contour_row in contour_dataframe.iterrows():
        color, _ = _mesoscale_eddy_marker_style(contour_row[eddies.POLARITY_COLUMN])
        axis.plot(
            contour_row[eddies.CONTOUR_LONGITUDES_COLUMN],
            contour_row[eddies.CONTOUR_LATITUDES_COLUMN],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )


def _plot_mesoscale_eddy_markers(axis, detections: xarray.DataArray | xarray.Dataset | object) -> None:
    if len(detections) == 0:
        return
    for polarity in eddies.POLARITY_ORDER:
        subset = detections.loc[detections[eddies.POLARITY_COLUMN] == polarity]
        if subset.empty:
            continue
        color, marker = _mesoscale_eddy_marker_style(polarity)
        axis.scatter(
            subset[Dimension.LONGITUDE.key()],
            subset[Dimension.LATITUDE.key()],
            c=color,
            marker=marker,
            s=22,
            linewidths=0.4,
            edgecolors="white",
            label=eddies.POLARITY_LABELS[polarity],
        )


def plot_mesoscale_eddy_overlay_gallery(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    reference_name: str = EDDY_REFERENCE_NAME,
    first_day_index: int = 0,
    lead_day_indices: Sequence[int] = DEFAULT_EDDY_LEAD_DAY_INDICES,
    challenger_detections=None,
    reference_detections=None,
    challenger_contours=None,
    reference_contours=None,
    show_markers: bool = True,
    show_contours: bool = True,
    **detection_parameters,
):
    resolved_detection_parameters = {
        **eddies.default_eddy_detection_parameters(),
        **detection_parameters,
    }
    if challenger_detections is None:
        challenger_detections = eddies.detect_mesoscale_eddies(
            challenger_dataset,
            first_day_index=first_day_index,
            lead_day_indices=list(lead_day_indices),
            background_sigma_grid=resolved_detection_parameters["background_sigma_grid"],
            detection_sigma_grid=resolved_detection_parameters["detection_sigma_grid"],
            min_distance_grid=resolved_detection_parameters["min_distance_grid"],
            amplitude_threshold_meters=resolved_detection_parameters["amplitude_threshold_meters"],
            max_abs_latitude_degrees=resolved_detection_parameters["max_abs_latitude_degrees"],
        )
    if reference_detections is None:
        reference_detections = eddies.detect_mesoscale_eddies(
            reference_dataset,
            first_day_index=first_day_index,
            lead_day_indices=list(lead_day_indices),
            background_sigma_grid=resolved_detection_parameters["background_sigma_grid"],
            detection_sigma_grid=resolved_detection_parameters["detection_sigma_grid"],
            min_distance_grid=resolved_detection_parameters["min_distance_grid"],
            amplitude_threshold_meters=resolved_detection_parameters["amplitude_threshold_meters"],
            max_abs_latitude_degrees=resolved_detection_parameters["max_abs_latitude_degrees"],
        )
    if challenger_contours is None:
        challenger_contours = eddies.mesoscale_eddy_contours_from_detections(
            challenger_detections,
            challenger_dataset,
            first_day_index=first_day_index,
            background_sigma_grid=resolved_detection_parameters["background_sigma_grid"],
            detection_sigma_grid=resolved_detection_parameters["detection_sigma_grid"],
            amplitude_threshold_meters=resolved_detection_parameters["amplitude_threshold_meters"],
            max_abs_latitude_degrees=resolved_detection_parameters["max_abs_latitude_degrees"],
            contour_level_step_meters=resolved_detection_parameters["contour_level_step_meters"],
            min_contour_pixel_count=resolved_detection_parameters["min_contour_pixel_count"],
            max_contour_pixel_count=resolved_detection_parameters["max_contour_pixel_count"],
            min_contour_convexity=resolved_detection_parameters["min_contour_convexity"],
        )
    if reference_contours is None:
        reference_contours = eddies.mesoscale_eddy_contours_from_detections(
            reference_detections,
            reference_dataset,
            first_day_index=first_day_index,
            background_sigma_grid=resolved_detection_parameters["background_sigma_grid"],
            detection_sigma_grid=resolved_detection_parameters["detection_sigma_grid"],
            amplitude_threshold_meters=resolved_detection_parameters["amplitude_threshold_meters"],
            max_abs_latitude_degrees=resolved_detection_parameters["max_abs_latitude_degrees"],
            contour_level_step_meters=resolved_detection_parameters["contour_level_step_meters"],
            min_contour_pixel_count=resolved_detection_parameters["min_contour_pixel_count"],
            max_contour_pixel_count=resolved_detection_parameters["max_contour_pixel_count"],
            min_contour_convexity=resolved_detection_parameters["min_contour_convexity"],
        )
    challenger_detections = eddies.filter_mesoscale_eddy_detections_by_contours(
        challenger_detections,
        challenger_contours,
    )
    reference_detections = eddies.filter_mesoscale_eddy_detections_by_contours(
        reference_detections,
        reference_contours,
    )
    figure, axes = plt.subplots(
        len(lead_day_indices),
        2,
        figsize=(16.0, 6.2 * len(lead_day_indices)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_index, lead_day_index in enumerate(lead_day_indices):
        challenger_field = eddies.surface_ssh_field(
            challenger_dataset,
            first_day_index=first_day_index,
            lead_day_index=lead_day_index,
        )
        reference_field = eddies.surface_ssh_field(
            reference_dataset,
            first_day_index=first_day_index,
            lead_day_index=lead_day_index,
        )
        cmap, norm = _field_norm(
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
            [challenger_field, reference_field],
        )

        challenger_axis = axes[row_index, 0]
        challenger_mesh = _plot_map(
            challenger_axis,
            challenger_field,
            title=f"GLONET lead day {lead_day_index + 1}",
            cmap=cmap,
            norm=norm,
        )
        if show_contours:
            _plot_mesoscale_eddy_contours(
                challenger_axis,
                challenger_contours.loc[challenger_contours[eddies.LEAD_DAY_COLUMN] == lead_day_index],
            )
        if show_markers:
            _plot_mesoscale_eddy_markers(
                challenger_axis,
                challenger_detections.loc[challenger_detections[eddies.LEAD_DAY_COLUMN] == lead_day_index],
            )
            challenger_axis.legend(loc="lower left")

        reference_axis = axes[row_index, 1]
        reference_mesh = _plot_map(
            reference_axis,
            reference_field,
            title=f"{reference_name} lead day {lead_day_index + 1}",
            cmap=cmap,
            norm=norm,
        )
        if show_contours:
            _plot_mesoscale_eddy_contours(
                reference_axis,
                reference_contours.loc[reference_contours[eddies.LEAD_DAY_COLUMN] == lead_day_index],
            )
        if show_markers:
            _plot_mesoscale_eddy_markers(
                reference_axis,
                reference_detections.loc[reference_detections[eddies.LEAD_DAY_COLUMN] == lead_day_index],
            )
            reference_axis.legend(loc="lower left")

        figure.colorbar(
            reference_mesh if row_index else challenger_mesh,
            ax=[challenger_axis, reference_axis],
            shrink=0.9,
            pad=0.01,
            label="Sea surface height (m)",
        )

    figure.suptitle(f"Accepted mesoscale eddies against {reference_name}", fontsize=16)
    return figure


def plot_mesoscale_eddy_concentration_gallery(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    reference_name: str = EDDY_REFERENCE_NAME,
    first_day_index: int = 0,
    challenger_detections=None,
    reference_detections=None,
    challenger_contours=None,
    reference_contours=None,
    **detection_parameters,
):
    resolved_detection_parameters = {
        **eddies.default_eddy_detection_parameters(),
        **detection_parameters,
    }
    if challenger_detections is None:
        challenger_detections = eddies.detect_mesoscale_eddies(
            challenger_dataset,
            first_day_index=first_day_index,
            background_sigma_grid=resolved_detection_parameters["background_sigma_grid"],
            detection_sigma_grid=resolved_detection_parameters["detection_sigma_grid"],
            min_distance_grid=resolved_detection_parameters["min_distance_grid"],
            amplitude_threshold_meters=resolved_detection_parameters["amplitude_threshold_meters"],
            max_abs_latitude_degrees=resolved_detection_parameters["max_abs_latitude_degrees"],
        )
    if challenger_contours is None:
        challenger_contours = eddies.mesoscale_eddy_contours_from_detections(
            challenger_detections,
            challenger_dataset,
            first_day_index=first_day_index,
            background_sigma_grid=resolved_detection_parameters["background_sigma_grid"],
            detection_sigma_grid=resolved_detection_parameters["detection_sigma_grid"],
            amplitude_threshold_meters=resolved_detection_parameters["amplitude_threshold_meters"],
            max_abs_latitude_degrees=resolved_detection_parameters["max_abs_latitude_degrees"],
            contour_level_step_meters=resolved_detection_parameters["contour_level_step_meters"],
            min_contour_pixel_count=resolved_detection_parameters["min_contour_pixel_count"],
            max_contour_pixel_count=resolved_detection_parameters["max_contour_pixel_count"],
            min_contour_convexity=resolved_detection_parameters["min_contour_convexity"],
        )
    challenger_detections = eddies.filter_mesoscale_eddy_detections_by_contours(
        challenger_detections,
        challenger_contours,
    )
    if reference_detections is None:
        reference_detections = eddies.detect_mesoscale_eddies(
            reference_dataset,
            first_day_index=first_day_index,
            background_sigma_grid=resolved_detection_parameters["background_sigma_grid"],
            detection_sigma_grid=resolved_detection_parameters["detection_sigma_grid"],
            min_distance_grid=resolved_detection_parameters["min_distance_grid"],
            amplitude_threshold_meters=resolved_detection_parameters["amplitude_threshold_meters"],
            max_abs_latitude_degrees=resolved_detection_parameters["max_abs_latitude_degrees"],
        )
    if reference_contours is None:
        reference_contours = eddies.mesoscale_eddy_contours_from_detections(
            reference_detections,
            reference_dataset,
            first_day_index=first_day_index,
            background_sigma_grid=resolved_detection_parameters["background_sigma_grid"],
            detection_sigma_grid=resolved_detection_parameters["detection_sigma_grid"],
            amplitude_threshold_meters=resolved_detection_parameters["amplitude_threshold_meters"],
            max_abs_latitude_degrees=resolved_detection_parameters["max_abs_latitude_degrees"],
            contour_level_step_meters=resolved_detection_parameters["contour_level_step_meters"],
            min_contour_pixel_count=resolved_detection_parameters["min_contour_pixel_count"],
            max_contour_pixel_count=resolved_detection_parameters["max_contour_pixel_count"],
            min_contour_convexity=resolved_detection_parameters["min_contour_convexity"],
        )
    reference_detections = eddies.filter_mesoscale_eddy_detections_by_contours(
        reference_detections,
        reference_contours,
    )
    challenger_concentration = eddies.mesoscale_eddy_concentration_from_contours(
        challenger_contours,
        challenger_dataset,
        first_day_index=first_day_index,
    )
    reference_concentration = eddies.mesoscale_eddy_concentration_from_contours(
        reference_contours,
        reference_dataset,
        first_day_index=first_day_index,
    )
    all_values = numpy.concatenate(
        [
            challenger_concentration[f"{polarity}_concentration"].values.ravel()
            for polarity in eddies.POLARITY_ORDER
        ]
        + [
            reference_concentration[f"{polarity}_concentration"].values.ravel()
            for polarity in eddies.POLARITY_ORDER
        ]
    )
    vmax = float(numpy.nanmax(all_values)) if numpy.any(numpy.isfinite(all_values)) else 1.0
    if vmax <= 0:
        vmax = 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15.0, 10.0),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (dataset_name, concentration_dataset) in enumerate(
        [
            ("GLONET", challenger_concentration),
            (reference_name, reference_concentration),
        ]
    ):
        for column_index, polarity in enumerate(eddies.POLARITY_ORDER):
            axis = axes[row_index, column_index]
            field = concentration_dataset[f"{polarity}_concentration"].copy()
            field.values = numpy.where(field.values > 0, field.values, numpy.nan)
            mesh = axis.pcolormesh(
                field[Dimension.LONGITUDE.key()].values,
                field[Dimension.LATITUDE.key()].values,
                field.values,
                shading="auto",
                cmap="viridis",
                norm=norm,
            )
            axis.set_title(f"{dataset_name} {eddies.POLARITY_LABELS[polarity].lower()}")
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
            figure.colorbar(mesh, ax=axis, shrink=0.9, pad=0.01, label="Detected eddy count")

    figure.suptitle("Accepted mesoscale eddy concentration over all lead days", fontsize=16)
    return figure


def _preferred_class4_depth_bin(variable_key: str, subset) -> str:
    preferred_depth_bin = DEFAULT_CLASS4_DEPTH_BINS.get(variable_key)
    if preferred_depth_bin and not subset.loc[subset["depth_bin"] == preferred_depth_bin].empty:
        return preferred_depth_bin
    return subset["depth_bin"].value_counts().idxmax()


def _class4_depth_selector(variable_key: str, depth_bin: str) -> float | None:
    if depth_bin in {"surface", "SST"}:
        return None
    depth_bins = DEPTH_BINS_BY_VARIABLE.get(variable_key, DEPTH_BINS_DEFAULT)
    if depth_bin not in depth_bins:
        return None
    lower_bound, upper_bound = depth_bins[depth_bin]
    return 0.5 * (max(lower_bound, 0.0) + upper_bound)


def _nearest_coordinate_indices(
    coordinates: numpy.ndarray,
    values: numpy.ndarray,
) -> numpy.ndarray:
    insertion_indices = numpy.searchsorted(coordinates, values)
    insertion_indices = numpy.clip(insertion_indices, 1, len(coordinates) - 1)
    left_values = coordinates[insertion_indices - 1]
    right_values = coordinates[insertion_indices]
    choose_left = numpy.abs(values - left_values) <= numpy.abs(right_values - values)
    return insertion_indices - choose_left.astype(int)


def _bin_class4_errors_to_model_grid(
    challenger_dataset: xarray.Dataset,
    subset,
    variable_key: str,
    lead_day: int,
    depth_bin: str,
) -> xarray.DataArray:
    model_field = _select_map_field(
        challenger_dataset,
        variable_key,
        first_day_index=0,
        lead_day_index=lead_day,
        depth_selector=_class4_depth_selector(variable_key, depth_bin),
    )
    pixelized_values = numpy.full(model_field.shape, numpy.nan, dtype=float)
    observation_counts = numpy.zeros(model_field.shape, dtype=int)
    latitude_values = model_field[Dimension.LATITUDE.key()].values
    longitude_values = _wrap_longitudes(model_field[Dimension.LONGITUDE.key()].values)
    observation_latitudes = subset[Dimension.LATITUDE.key()].to_numpy()
    observation_longitudes = _wrap_longitudes(subset[Dimension.LONGITUDE.key()].to_numpy())
    absolute_errors = subset["absolute_error"].to_numpy()
    valid_mask = (
        numpy.isfinite(observation_latitudes)
        & numpy.isfinite(observation_longitudes)
        & numpy.isfinite(absolute_errors)
    )
    if not numpy.any(valid_mask):
        return xarray.DataArray(
            pixelized_values,
            dims=model_field.dims,
            coords=model_field.coords,
        )

    latitude_indices = _nearest_coordinate_indices(latitude_values, observation_latitudes[valid_mask])
    longitude_indices = _nearest_coordinate_indices(longitude_values, observation_longitudes[valid_mask])
    valid_errors = absolute_errors[valid_mask]

    for latitude_index, longitude_index, absolute_error in zip(
        latitude_indices,
        longitude_indices,
        valid_errors,
        strict=False,
    ):
        if not numpy.isfinite(model_field.values[latitude_index, longitude_index]):
            continue
        if observation_counts[latitude_index, longitude_index] == 0:
            pixelized_values[latitude_index, longitude_index] = absolute_error
        else:
            current_count = observation_counts[latitude_index, longitude_index]
            pixelized_values[latitude_index, longitude_index] = (
                pixelized_values[latitude_index, longitude_index] * current_count + absolute_error
            ) / (current_count + 1)
        observation_counts[latitude_index, longitude_index] += 1

    return xarray.DataArray(
        pixelized_values,
        dims=model_field.dims,
        coords=model_field.coords,
    )


def _plot_class4_error_map(
    challenger_dataset: xarray.Dataset,
    axis,
    subset,
    variable_key: str,
    lead_day: int,
    depth_bin: str,
    norm: Normalize,
    render_mode: str,
    grid_resolution_degrees: float,
    interpolation_method: str,
):
    if grid_resolution_degrees <= 0:
        raise ValueError("grid_resolution_degrees must be positive")
    if render_mode not in {"scatter", "grid", "pixel"}:
        raise ValueError("render_mode must be one of 'scatter', 'grid', or 'pixel'")

    if render_mode in {"grid", "pixel"}:
        pixelized_field = _bin_class4_errors_to_model_grid(
            challenger_dataset=challenger_dataset,
            subset=subset,
            variable_key=variable_key,
            lead_day=lead_day,
            depth_bin=depth_bin,
        )
        return axis.pcolormesh(
            _wrap_longitudes(pixelized_field[Dimension.LONGITUDE.key()].values),
            pixelized_field[Dimension.LATITUDE.key()].values,
            pixelized_field.values,
            shading="auto",
            cmap=ERROR_CMAP,
            norm=norm,
        )

    return axis.scatter(
        _wrap_longitudes(subset[Dimension.LONGITUDE.key()].to_numpy()),
        subset[Dimension.LATITUDE.key()],
        c=subset["absolute_error"],
        s=5,
        cmap=ERROR_CMAP,
        norm=norm,
        linewidths=0,
        alpha=0.75,
    )


def plot_class4_scatter_gallery(
    challenger_dataset: xarray.Dataset,
    observation_dataset: xarray.Dataset,
    variables: Sequence[Variable | str] = DEFAULT_VARIABLES,
    lead_day: int = 0,
    render_mode: str = "scatter",
    grid_resolution_degrees: float = DEFAULT_SCATTER_GRID_RESOLUTION_DEGREES,
    interpolation_method: str = "linear",
):
    variable_enums = [_as_variable_enum(variable) for variable in variables]
    minimum_longitude, maximum_longitude, minimum_latitude, maximum_latitude = _domain_bounds(challenger_dataset)
    comparison_dataframe = classIV.class4_validation_dataframe(
        challenger_dataset=challenger_dataset,
        reference_dataset=observation_dataset,
        variables=variable_enums,
    )

    figure, axes = plt.subplots(
        math.ceil(len(variable_enums) / 2),
        2,
        figsize=(12, 4.2 * math.ceil(len(variable_enums) / 2)),
        squeeze=False,
        constrained_layout=True,
    )
    flattened_axes = axes.flatten()

    for axis in flattened_axes[len(variable_enums) :]:
        axis.axis("off")

    for index, variable in enumerate(variable_enums):
        axis = flattened_axes[index]
        variable_key = variable.key()
        subset = comparison_dataframe.loc[
            (comparison_dataframe["variable"] == variable_key) & (comparison_dataframe["lead_day"] == lead_day)
        ].copy()
        if subset.empty:
            axis.set_axis_off()
            continue
        depth_bin = _preferred_class4_depth_bin(variable_key, subset)
        subset = subset.loc[subset["depth_bin"] == depth_bin]
        norm = _positive_error_norm(subset["absolute_error"].to_numpy())
        scatter_or_mesh = _plot_class4_error_map(
            challenger_dataset=challenger_dataset,
            axis=axis,
            subset=subset,
            variable_key=variable_key,
            lead_day=lead_day,
            depth_bin=depth_bin,
            norm=norm,
            render_mode=render_mode,
            grid_resolution_degrees=grid_resolution_degrees,
            interpolation_method=interpolation_method,
        )
        axis.set_title(f"{_friendly_variable_label(variable_key)} ({depth_bin})")
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.set_xlim(minimum_longitude, maximum_longitude)
        axis.set_ylim(minimum_latitude, maximum_latitude)
        figure.colorbar(scatter_or_mesh, ax=axis, shrink=0.9, pad=0.01, label="Absolute error")

    view_label = "pixelized" if render_mode in {"grid", "pixel"} else "scatter"
    figure.suptitle(
        f"Class-4 observation absolute error {view_label} maps for lead day {lead_day + 1}",
        fontsize=16,
    )
    return figure


def _trajectory_particle_distances(
    challenger_particles: xarray.Dataset,
    reference_particles: xarray.Dataset,
) -> numpy.ndarray:
    challenger_particles, reference_particles = xarray.align(challenger_particles, reference_particles, join="inner")
    latitude_reference_set_rad = numpy.deg2rad(reference_particles["lat"].values)
    dlatitude = (challenger_particles["lat"].values - reference_particles["lat"].values) * 111
    dlongitude = (
        (challenger_particles["lon"].values - reference_particles["lon"].values)
        * 111
        * numpy.cos(latitude_reference_set_rad)
    )
    return numpy.sqrt(dlatitude**2 + dlongitude**2)


def _sample_particle_indices(
    particle_count: int,
    particle_percentage: float,
    seed: int,
) -> numpy.ndarray:
    if particle_percentage < 10.0 or particle_percentage > 100.0:
        raise ValueError("particle_percentage must be between 10 and 100")
    displayed_particle_count = max(1, math.ceil(particle_count * particle_percentage / 100.0))
    random_generator = numpy.random.default_rng(seed)
    sampled_indices = random_generator.choice(particle_count, size=displayed_particle_count, replace=False)
    return numpy.sort(sampled_indices)


def _final_finite_distances(particle_distances: numpy.ndarray) -> numpy.ndarray:
    valid_mask = numpy.isfinite(particle_distances)
    reversed_valid_mask = valid_mask[:, ::-1]
    has_valid_distance = reversed_valid_mask.any(axis=1)
    final_indices = particle_distances.shape[1] - 1 - reversed_valid_mask.argmax(axis=1)
    final_distances = numpy.full(particle_distances.shape[0], numpy.nan)
    valid_particle_indices = numpy.where(has_valid_distance)[0]
    final_distances[valid_particle_indices] = particle_distances[
        valid_particle_indices,
        final_indices[valid_particle_indices],
    ]
    return final_distances


def _plot_periodic_trajectory(
    ax,
    longitudes: numpy.ndarray,
    latitudes: numpy.ndarray,
    **plot_kwargs,
):
    wrapped_longitudes = _wrap_longitudes(numpy.asarray(longitudes, dtype=float))
    latitudes = numpy.asarray(latitudes, dtype=float)
    current_segment_longitudes: list[float] = []
    current_segment_latitudes: list[float] = []

    def flush_segment():
        if len(current_segment_longitudes) >= 2:
            ax.plot(current_segment_longitudes, current_segment_latitudes, **plot_kwargs)

    for longitude, latitude in zip(wrapped_longitudes, latitudes, strict=False):
        if not numpy.isfinite(longitude) or not numpy.isfinite(latitude):
            flush_segment()
            current_segment_longitudes = []
            current_segment_latitudes = []
            continue

        if not current_segment_longitudes:
            current_segment_longitudes = [float(longitude)]
            current_segment_latitudes = [float(latitude)]
            continue

        previous_longitude = current_segment_longitudes[-1]
        previous_latitude = current_segment_latitudes[-1]
        longitude_delta = float(longitude - previous_longitude)
        if longitude_delta > LONGITUDE_SPAN / 2:
            adjusted_longitude = float(longitude - LONGITUDE_SPAN)
            boundary_longitude = LONGITUDE_MIN
            restart_longitude = LONGITUDE_MAX
        elif longitude_delta < -LONGITUDE_SPAN / 2:
            adjusted_longitude = float(longitude + LONGITUDE_SPAN)
            boundary_longitude = LONGITUDE_MAX
            restart_longitude = LONGITUDE_MIN
        else:
            current_segment_longitudes.append(float(longitude))
            current_segment_latitudes.append(float(latitude))
            continue

        if numpy.isclose(adjusted_longitude, previous_longitude):
            boundary_latitude = 0.5 * (previous_latitude + latitude)
        else:
            boundary_fraction = (boundary_longitude - previous_longitude) / (adjusted_longitude - previous_longitude)
            boundary_latitude = previous_latitude + boundary_fraction * (latitude - previous_latitude)
        current_segment_longitudes.append(float(boundary_longitude))
        current_segment_latitudes.append(float(boundary_latitude))
        flush_segment()
        current_segment_longitudes = [float(restart_longitude), float(longitude)]
        current_segment_latitudes = [float(boundary_latitude), float(latitude)]

    flush_segment()


def _plot_trajectory_dataset_comparison(
    challenger_particles: xarray.Dataset,
    reference_particles: xarray.Dataset,
    reference_name: str,
    particle_percentage: float,
    seed: int,
    minimum_longitude: float,
    maximum_longitude: float,
    minimum_latitude: float,
    maximum_latitude: float,
    title_particle_count: int | None = None,
    resample_particles: bool = True,
):
    challenger_particles, reference_particles = xarray.align(challenger_particles, reference_particles, join="inner")
    particle_distances = _trajectory_particle_distances(challenger_particles, reference_particles)
    if resample_particles:
        sampled_indices = _sample_particle_indices(
            particle_count=reference_particles.sizes["particle"],
            particle_percentage=particle_percentage,
            seed=seed,
        )
        sampled_reference_particles = reference_particles.isel(particle=sampled_indices)
        sampled_challenger_particles = challenger_particles.isel(particle=sampled_indices)
        sampled_distances = particle_distances[sampled_indices]
    else:
        sampled_reference_particles = reference_particles
        sampled_challenger_particles = challenger_particles
        sampled_distances = particle_distances
    final_particle_distance = _final_finite_distances(sampled_distances)
    norm = _positive_error_norm(final_particle_distance)
    figure, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)

    for particle_index in range(sampled_reference_particles.sizes["particle"]):
        _plot_periodic_trajectory(
            ax,
            sampled_reference_particles["lon"].values[particle_index],
            sampled_reference_particles["lat"].values[particle_index],
            color="lightgray",
            linewidth=0.8,
            alpha=0.45,
        )
        _plot_periodic_trajectory(
            ax,
            sampled_challenger_particles["lon"].values[particle_index],
            sampled_challenger_particles["lat"].values[particle_index],
            color=plt.get_cmap(ERROR_CMAP)(norm(final_particle_distance[particle_index])),
            linewidth=1.0,
            alpha=0.85,
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.scatter(
        _wrap_longitudes(sampled_reference_particles["lon0"].values),
        sampled_reference_particles["lat0"].values,
        s=3,
        color="black",
        alpha=0.4,
        label="Initial particles",
    )
    ax.set_xlim(minimum_longitude, maximum_longitude)
    ax.set_ylim(minimum_latitude, maximum_latitude)
    ax.legend(loc="lower left")
    figure.colorbar(
        ScalarMappable(norm=norm, cmap=ERROR_CMAP),
        ax=ax,
        shrink=0.9,
        pad=0.01,
        label="Final separation (km)",
    )
    displayed_particle_count = title_particle_count or reference_particles.sizes["particle"]
    ax.set_title(
        f"Lagrangian trajectories vs {reference_name} "
        f"({particle_percentage:.0f}% of {displayed_particle_count} trajectories)"
    )
    return figure


def plot_lagrangian_trajectory_comparison(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    reference_name: str,
    first_day_index: int = 0,
    particle_percentage: float = 10.0,
    seed: int = 123,
):
    minimum_longitude, maximum_longitude, minimum_latitude, maximum_latitude = _domain_bounds(challenger_dataset)
    challenger_standard_dataset = lagrangian_trajectory._harmonise_dataset(challenger_dataset)
    reference_standard_dataset = lagrangian_trajectory._harmonise_dataset(reference_dataset)
    initial_latitudes, initial_longitudes = lagrangian_trajectory._get_random_ocean_points_from_file(
        challenger_standard_dataset,
        variable_name=Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
        n=lagrangian_trajectory.INITIAL_PARTICLE_COUNT,
        seed=seed,
    )
    sampled_indices = _sample_particle_indices(
        particle_count=len(initial_latitudes),
        particle_percentage=particle_percentage,
        seed=seed,
    )
    latitudes = initial_latitudes[sampled_indices]
    longitudes = initial_longitudes[sampled_indices]

    challenger_run_dataset = lagrangian_trajectory._split_dataset(challenger_standard_dataset)[first_day_index]
    reference_run_dataset = lagrangian_trajectory._split_dataset(reference_standard_dataset)[first_day_index]

    challenger_particles = lagrangian_trajectory._get_particle_dataset(
        dataset=challenger_run_dataset.isel({Dimension.DEPTH.key(): 0}),
        latitudes=latitudes,
        longitudes=longitudes,
    )
    reference_particles = lagrangian_trajectory._get_particle_dataset(
        dataset=reference_run_dataset.isel({Dimension.DEPTH.key(): 0}),
        latitudes=latitudes,
        longitudes=longitudes,
    )
    figure = _plot_trajectory_dataset_comparison(
        challenger_particles=challenger_particles,
        reference_particles=reference_particles,
        reference_name=reference_name,
        particle_percentage=particle_percentage,
        seed=seed,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        title_particle_count=len(initial_latitudes),
        resample_particles=False,
    )
    return figure


def plot_class4_drifter_trajectory_comparison(
    challenger_dataset: xarray.Dataset,
    observation_dataset: xarray.Dataset,
    first_day_index: int = 0,
    particle_percentage: float = 20.0,
    seed: int = 123,
):
    minimum_longitude, maximum_longitude, minimum_latitude, maximum_latitude = _domain_bounds(challenger_dataset)
    challenger_particles, reference_particles = class4_drifters.class4_drifter_trajectory_comparison(
        challenger_dataset=challenger_dataset,
        observation_dataset=observation_dataset,
        first_day_index=first_day_index,
    )
    return _plot_trajectory_dataset_comparison(
        challenger_particles=challenger_particles,
        reference_particles=reference_particles,
        reference_name="Class-4 drifter observations",
        particle_percentage=particle_percentage,
        seed=seed,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        title_particle_count=reference_particles.sizes["particle"],
    )
