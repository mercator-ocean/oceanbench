# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Mapping, Sequence
from base64 import b64encode
from dataclasses import dataclass
import html
from io import BytesIO
import json
import warnings
from uuid import uuid4

from dask import compute as dask_compute
from IPython.display import HTML
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy
import pandas
import xarray

from oceanbench.core import eddies
from oceanbench.core import lagrangian_trajectory
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import DepthLevel, Dimension, Variable, VARIABLE_LABELS, VARIABLE_METADATA

DEFAULT_SURFACE_COMPARISON_VARIABLES: tuple[Variable, ...] = (
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
    Variable.SEA_WATER_SALINITY,
    Variable.EASTWARD_SEA_WATER_VELOCITY,
    Variable.NORTHWARD_SEA_WATER_VELOCITY,
)
DEFAULT_ZONAL_PSD_VARIABLES: tuple[Variable, ...] = (
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
)
DEFAULT_ZONAL_PSD_LEAD_DAY_INDICES: tuple[int, ...] = (0, -1)
FIELD_COLORMAPS: dict[str, str] = {
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): "RdBu_r",
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): "viridis",
    Variable.SEA_WATER_SALINITY.key(): "cividis",
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(): "RdBu_r",
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): "RdBu_r",
    Variable.MIXED_LAYER_DEPTH.key(): "viridis",
    Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY.key(): "RdBu_r",
    Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY.key(): "RdBu_r",
}
DIVERGING_VARIABLES = {
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
    Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY.key(),
    Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY.key(),
}
ERROR_COLORMAP = "RdBu_r"
ABSOLUTE_ERROR_COLORMAP = "magma"
RMSE_COLORMAP = "magma"
LAND_BACKGROUND_COLOR = "#d9d5cc"
NO_DATA_COLOR = "#eef2f7"
DEFAULT_EXPLORER_MAXIMUM_MAP_CELLS = 160_000
DEFAULT_EXPLORER_HEIGHT_PIXELS = 760
DEFAULT_EXPLORER_IMAGE_QUALITY = 85
DEFAULT_LAGRANGIAN_PARTICLE_COUNT = 300
DEFAULT_LAGRANGIAN_MAXIMUM_LAND_MASK_CELLS = 80_000
DEFAULT_EDDY_MAXIMUM_CONTOUR_POINTS = 80
DEFAULT_SURFACE_COMPARISON_DEPTH_SELECTORS: tuple[float, ...] = tuple(depth_level.value for depth_level in DepthLevel)
GLOBAL_LONGITUDE_SPAN_THRESHOLD_DEGREES = 300.0


@dataclass(frozen=True)
class _ComputedReferenceComparisonFields:
    key: str
    label: str
    reference_fields: tuple[xarray.DataArray, ...]
    rmse_fields: tuple[xarray.DataArray, ...]


@dataclass(frozen=True)
class _ComputedMultiReferenceComparisonFields:
    lead_day_indices: tuple[int, ...]
    challenger_fields: tuple[xarray.DataArray, ...]
    references: tuple[_ComputedReferenceComparisonFields, ...]
    spatial_stride: int


def _as_variable_key(variable: Variable | str) -> str:
    return variable.key() if isinstance(variable, Variable) else variable


def _standard_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    return rename_dataset_with_standard_names(dataset)


def _variable_label(variable_key: str) -> str:
    return VARIABLE_LABELS[variable_key].capitalize()


def _variable_label_with_unit(variable_key: str) -> str:
    display_name, unit = VARIABLE_METADATA[variable_key]
    return f"{display_name.capitalize()} ({unit})"


def _select_dimension_index(
    data_array: xarray.DataArray,
    dimension_name: str,
    index: int,
) -> xarray.DataArray:
    if dimension_name not in data_array.dims:
        if index == 0:
            return data_array
        raise ValueError(f"Cannot select index {index} because dimension {dimension_name!r} is missing.")
    if index < 0 or index >= data_array.sizes[dimension_name]:
        raise ValueError(
            f"Index {index} is out of bounds for dimension {dimension_name!r} "
            + f"with size {data_array.sizes[dimension_name]}."
        )
    return data_array.isel({dimension_name: index})


def _select_depth(
    data_array: xarray.DataArray,
    depth_selector: int | float | None,
) -> xarray.DataArray:
    if Dimension.DEPTH.key() not in data_array.dims:
        return data_array
    if depth_selector is None:
        return data_array.isel({Dimension.DEPTH.key(): 0})
    if isinstance(depth_selector, int):
        return _select_dimension_index(data_array, Dimension.DEPTH.key(), depth_selector)
    return data_array.sel({Dimension.DEPTH.key(): depth_selector}, method="nearest")


def _select_variable_field(
    dataset: xarray.Dataset,
    variable_key: str,
    depth_selector: int | float | None,
) -> xarray.DataArray:
    standard_dataset = _standard_dataset(dataset)
    if variable_key not in standard_dataset:
        raise ValueError(f"Dataset does not contain variable {variable_key!r}.")

    field = standard_dataset[variable_key]
    return _select_depth(field, depth_selector)


def _select_lead_day_indices(
    data_array: xarray.DataArray,
    lead_day_indices: Sequence[int],
) -> xarray.DataArray:
    if Dimension.LEAD_DAY_INDEX.key() not in data_array.dims:
        if tuple(lead_day_indices) == (0,):
            return data_array
        raise ValueError(
            f"Cannot select lead day indices {tuple(lead_day_indices)!r} "
            + f"because dimension {Dimension.LEAD_DAY_INDEX.key()!r} is missing."
        )
    return data_array.isel({Dimension.LEAD_DAY_INDEX.key(): list(lead_day_indices)})


def _select_first_day_index(
    data_array: xarray.DataArray,
    first_day_index: int,
) -> xarray.DataArray:
    return _select_dimension_index(
        data_array,
        Dimension.FIRST_DAY_DATETIME.key(),
        first_day_index,
    )


def _depth_label(field: xarray.DataArray) -> str:
    if Dimension.DEPTH.key() not in field.coords:
        return "Surface"
    depth_value = float(field[Dimension.DEPTH.key()].item())
    if abs(depth_value) < 10.0:
        return f"{depth_value:.1f} m"
    return f"{depth_value:.0f} m"


def _finite_values(arrays: Sequence[xarray.DataArray]) -> numpy.ndarray:
    values = [numpy.asarray(array.values, dtype=float).ravel() for array in arrays]
    if not values:
        return numpy.array([])
    flattened_values = numpy.concatenate(values)
    return flattened_values[numpy.isfinite(flattened_values)]


def _positive_norm(arrays: Sequence[xarray.DataArray]) -> Normalize:
    finite_values = _finite_values(arrays)
    if finite_values.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    minimum = float(numpy.nanmin(finite_values))
    maximum = float(numpy.nanmax(finite_values))
    if maximum <= minimum:
        maximum = minimum + 1.0
    return Normalize(vmin=minimum, vmax=maximum)


def _symmetric_norm(arrays: Sequence[xarray.DataArray]) -> Normalize:
    finite_values = _finite_values(arrays)
    if finite_values.size == 0:
        return Normalize(vmin=-1.0, vmax=1.0)
    maximum_absolute_value = float(numpy.nanmax(numpy.abs(finite_values)))
    if maximum_absolute_value == 0.0:
        maximum_absolute_value = 1.0
    return TwoSlopeNorm(vcenter=0.0, vmin=-maximum_absolute_value, vmax=maximum_absolute_value)


def _field_norm(variable_key: str, fields: Sequence[xarray.DataArray]) -> Normalize:
    if variable_key in DIVERGING_VARIABLES:
        return _symmetric_norm(fields)
    return _positive_norm(fields)


def _field_colormap(variable_key: str) -> str:
    if variable_key not in FIELD_COLORMAPS:
        raise ValueError(f"No surface comparison colormap is configured for variable {variable_key!r}.")
    return FIELD_COLORMAPS[variable_key]


def _lead_day_label(field: xarray.DataArray, lead_day_index: int) -> str:
    if Dimension.LEAD_DAY_INDEX.key() not in field.coords:
        return str(lead_day_index + 1)
    return str(int(field[Dimension.LEAD_DAY_INDEX.key()].item()) + 1)


def _all_lead_day_indices(dataset: xarray.Dataset) -> tuple[int, ...]:
    standard_dataset = _standard_dataset(dataset)
    lead_day_count = standard_dataset.sizes.get(Dimension.LEAD_DAY_INDEX.key(), 1)
    return tuple(range(lead_day_count))


def _resolved_lead_day_indices(dataset: xarray.Dataset, lead_day_indices: Sequence[int]) -> tuple[int, ...]:
    all_lead_day_indices = _all_lead_day_indices(dataset)
    resolved_indices = []
    for lead_day_index in lead_day_indices:
        resolved_index = all_lead_day_indices[lead_day_index] if lead_day_index < 0 else lead_day_index
        if resolved_index in all_lead_day_indices:
            resolved_indices.append(resolved_index)
    return tuple(dict.fromkeys(resolved_indices))


def _reference_items(
    reference_datasets: Mapping[str, xarray.Dataset],
) -> tuple[tuple[str, str, xarray.Dataset], ...]:
    if not reference_datasets:
        raise ValueError("reference_datasets must contain at least one reference dataset.")
    return tuple(
        (f"reference_{index}", reference_name, reference_dataset)
        for index, (reference_name, reference_dataset) in enumerate(reference_datasets.items())
    )


def _zonal_wavelength_kilometers(field: xarray.DataArray) -> numpy.ndarray:
    longitude = numpy.asarray(field[Dimension.LONGITUDE.key()].values, dtype=float)
    latitude = numpy.asarray(field[Dimension.LATITUDE.key()].values, dtype=float)
    longitude_spacing_degree = float(numpy.nanmedian(numpy.abs(numpy.diff(longitude))))
    mean_latitude_radians = numpy.deg2rad(float(numpy.nanmean(latitude)))
    longitude_spacing_meters = (
        longitude_spacing_degree * numpy.pi / 180.0 * 6_371_000.0 * numpy.cos(mean_latitude_radians)
    )
    frequencies = numpy.fft.rfftfreq(len(longitude), d=abs(longitude_spacing_meters))
    with numpy.errstate(divide="ignore"):
        wavelength_kilometers = 1.0 / frequencies / 1000.0
    return wavelength_kilometers


def _zonal_power_spectrum(field: xarray.DataArray) -> tuple[numpy.ndarray, numpy.ndarray]:
    values = numpy.asarray(field.values, dtype=float)
    finite_counts = numpy.sum(numpy.isfinite(values), axis=-1, keepdims=True)
    zonal_sums = numpy.nansum(values, axis=-1, keepdims=True)
    zonal_means = numpy.divide(zonal_sums, finite_counts, out=numpy.zeros_like(zonal_sums), where=finite_counts > 0)
    values = values - zonal_means
    values = numpy.nan_to_num(values, nan=0.0)
    spectrum = numpy.fft.rfft(values, axis=-1)
    power = numpy.nanmean(numpy.abs(spectrum) ** 2, axis=-2)
    wavelength_kilometers = _zonal_wavelength_kilometers(field)
    valid_mask = numpy.isfinite(wavelength_kilometers) & numpy.isfinite(power) & (wavelength_kilometers > 0)
    sorted_indices = numpy.argsort(wavelength_kilometers[valid_mask])
    return wavelength_kilometers[valid_mask][sorted_indices], power[valid_mask][sorted_indices]


def _zonal_psd_fields(
    challenger_dataset: xarray.Dataset,
    reference_datasets: Mapping[str, xarray.Dataset],
    variable_key: str,
    first_day_index: int,
    lead_day_indices: Sequence[int],
    depth_selector: int | float | None,
) -> tuple[xarray.DataArray, tuple[tuple[str, str, xarray.DataArray], ...]]:
    reference_items = _reference_items(reference_datasets)
    challenger_field = _select_lead_day_indices(
        _select_first_day_index(
            _select_variable_field(challenger_dataset, variable_key, depth_selector),
            first_day_index,
        ),
        lead_day_indices,
    )
    reference_fields = tuple(
        _select_lead_day_indices(
            _select_first_day_index(
                _select_variable_field(reference_dataset, variable_key, depth_selector),
                first_day_index,
            ),
            lead_day_indices,
        )
        for _, _, reference_dataset in reference_items
    )
    aligned_fields = dask_compute(*xarray.align(challenger_field, *reference_fields, join="inner"))
    return aligned_fields[0], tuple(
        (reference_key, reference_label, reference_field)
        for (reference_key, reference_label, _), reference_field in zip(
            reference_items,
            aligned_fields[1:],
            strict=True,
        )
    )


def _multi_reference_comparison_fields(
    challenger_dataset: xarray.Dataset,
    reference_datasets: Mapping[str, xarray.Dataset],
    variable_key: str,
    first_day_index: int,
    lead_day_indices: Sequence[int],
    depth_selector: int | float | None,
    maximum_map_cells: int,
) -> _ComputedMultiReferenceComparisonFields:
    reference_items = _reference_items(reference_datasets)
    challenger_field = _select_variable_field(
        challenger_dataset,
        variable_key,
        depth_selector,
    )
    reference_fields = tuple(
        _select_variable_field(reference_dataset, variable_key, depth_selector)
        for _, _, reference_dataset in reference_items
    )
    aligned_fields = xarray.align(challenger_field, *reference_fields, join="inner")
    challenger_field = aligned_fields[0]
    reference_fields = aligned_fields[1:]
    display_challenger_field = _select_lead_day_indices(
        _select_first_day_index(challenger_field, first_day_index),
        lead_day_indices,
    )
    spatial_stride = _spatial_stride(display_challenger_field, maximum_map_cells)
    thinned_display_challenger_field = _thin_field(display_challenger_field, spatial_stride)
    thinned_display_reference_fields = tuple(
        _thin_field(
            _select_lead_day_indices(_select_first_day_index(reference_field, first_day_index), lead_day_indices),
            spatial_stride,
        )
        for reference_field in reference_fields
    )
    thinned_rmse_fields = tuple(
        _rmse_over_forecast_dates_field(
            challenger_field,
            reference_field,
            lead_day_indices,
            spatial_stride,
        )
        for reference_field in reference_fields
    )
    computed_fields = dask_compute(
        thinned_display_challenger_field,
        *thinned_display_reference_fields,
        *thinned_rmse_fields,
    )
    computed_challenger_field = computed_fields[0]
    reference_count = len(reference_items)
    computed_reference_fields = computed_fields[1 : 1 + reference_count]
    computed_rmse_fields = computed_fields[1 + reference_count :]
    return _ComputedMultiReferenceComparisonFields(
        lead_day_indices=tuple(lead_day_indices),
        challenger_fields=_split_lead_day_fields(computed_challenger_field),
        references=tuple(
            _ComputedReferenceComparisonFields(
                key=reference_key,
                label=reference_name,
                reference_fields=_split_lead_day_fields(reference_field),
                rmse_fields=_split_lead_day_fields(rmse_field),
            )
            for (reference_key, reference_name, _), reference_field, rmse_field in zip(
                reference_items,
                computed_reference_fields,
                computed_rmse_fields,
                strict=True,
            )
        ),
        spatial_stride=spatial_stride,
    )


def _rmse_over_forecast_dates_field(
    challenger_field: xarray.DataArray,
    reference_field: xarray.DataArray,
    lead_day_indices: Sequence[int],
    spatial_stride: int,
) -> xarray.DataArray:
    thinned_challenger_field = _thin_field(
        _select_lead_day_indices(challenger_field, lead_day_indices),
        spatial_stride,
    )
    thinned_reference_field = _thin_field(
        _select_lead_day_indices(reference_field, lead_day_indices),
        spatial_stride,
    )
    error_field = thinned_challenger_field - thinned_reference_field
    reduction_dimensions = [
        Dimension.FIRST_DAY_DATETIME.key(),
    ]
    reduction_dimensions = [dimension for dimension in reduction_dimensions if dimension in error_field.dims]
    if not reduction_dimensions:
        return abs(error_field)
    return numpy.sqrt((error_field**2).mean(dim=reduction_dimensions))


def _split_lead_day_fields(data_array: xarray.DataArray) -> tuple[xarray.DataArray, ...]:
    if Dimension.LEAD_DAY_INDEX.key() not in data_array.dims:
        return (_visualization_field(data_array),)
    return tuple(
        _visualization_field(data_array.isel({Dimension.LEAD_DAY_INDEX.key(): lead_day_index}))
        for lead_day_index in range(data_array.sizes[Dimension.LEAD_DAY_INDEX.key()])
    )


def _visualization_field(data_array: xarray.DataArray) -> xarray.DataArray:
    if numpy.issubdtype(data_array.dtype, numpy.floating):
        return data_array.astype("float32", copy=False)
    return data_array


def _spatial_stride(field: xarray.DataArray, maximum_map_cells: int) -> int:
    latitude_count = field.sizes[Dimension.LATITUDE.key()]
    longitude_count = field.sizes[Dimension.LONGITUDE.key()]
    cell_count = latitude_count * longitude_count
    if cell_count <= maximum_map_cells:
        return 1
    return int(numpy.ceil(numpy.sqrt(cell_count / maximum_map_cells)))


def _thin_field(field: xarray.DataArray, stride: int) -> xarray.DataArray:
    if stride <= 1:
        return field
    return field.isel(
        {
            Dimension.LATITUDE.key(): slice(None, None, stride),
            Dimension.LONGITUDE.key(): slice(None, None, stride),
        }
    )


def _longitude_values_for_image(field: xarray.DataArray) -> numpy.ndarray:
    longitude = numpy.asarray(field[Dimension.LONGITUDE.key()].values, dtype=float)
    longitude_span = float(numpy.nanmax(longitude) - numpy.nanmin(longitude))
    if longitude_span >= GLOBAL_LONGITUDE_SPAN_THRESHOLD_DEGREES:
        return numpy.mod(longitude, 360.0)
    return longitude


def _image_extent(field: xarray.DataArray) -> tuple[float, float, float, float]:
    longitude = field[Dimension.LONGITUDE.key()].values
    latitude = field[Dimension.LATITUDE.key()].values
    return (
        float(numpy.nanmin(longitude)),
        float(numpy.nanmax(longitude)),
        float(numpy.nanmin(latitude)),
        float(numpy.nanmax(latitude)),
    )


def _prepared_image_field(field: xarray.DataArray) -> xarray.DataArray:
    image_longitude = _longitude_values_for_image(field)
    if numpy.array_equal(image_longitude, field[Dimension.LONGITUDE.key()].values):
        return field
    return field.assign_coords({Dimension.LONGITUDE.key(): image_longitude}).sortby(Dimension.LONGITUDE.key())


def _image_values(field: xarray.DataArray) -> numpy.ndarray:
    values = numpy.asarray(field.values, dtype=float)
    latitude = field[Dimension.LATITUDE.key()].values
    if len(latitude) > 1 and latitude[0] > latitude[-1]:
        return numpy.flipud(values)
    return values


def _add_geostrophic_equatorial_mask_band(axis, field: xarray.DataArray, variable_key: str) -> None:
    if variable_key not in {
        Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY.key(),
        Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY.key(),
    }:
        return
    latitude = numpy.asarray(field[Dimension.LATITUDE.key()].values, dtype=float)
    if numpy.nanmin(latitude) > 0.5 or numpy.nanmax(latitude) < -0.5:
        return
    axis.axhspan(
        -0.5,
        0.5,
        facecolor=NO_DATA_COLOR,
        edgecolor="#94a3b8",
        linewidth=0.5,
        alpha=0.96,
        zorder=3,
    )


def _encoded_webp_image(
    field: xarray.DataArray,
    norm: Normalize,
    colormap_name: str,
    value_label: str,
    title: str,
    variable_key: str,
) -> str:
    field = _prepared_image_field(field)
    colormap = colormaps[colormap_name].copy()
    colormap.set_bad(LAND_BACKGROUND_COLOR)
    figure, axis = plt.subplots(figsize=(11.0, 6.2), dpi=100)
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor(LAND_BACKGROUND_COLOR)
    image = axis.imshow(
        _image_values(field),
        cmap=colormap,
        extent=_image_extent(field),
        interpolation="nearest",
        norm=norm,
        origin="lower",
        aspect="equal",
    )
    _add_geostrophic_equatorial_mask_band(axis, field, variable_key)
    axis.set_title(title, fontsize=12)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(color="#334155", alpha=0.18, linewidth=0.6)
    colorbar = figure.colorbar(image, ax=axis, orientation="horizontal", pad=0.1, shrink=0.86)
    colorbar.set_label(value_label)
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="webp",
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
        pil_kwargs={"quality": DEFAULT_EXPLORER_IMAGE_QUALITY},
    )
    plt.close(figure)
    return "data:image/webp;base64," + b64encode(buffer.getvalue()).decode("ascii")


def _encoded_images(
    key: str,
    label: str,
    arrays: Sequence[xarray.DataArray],
    norm: Normalize,
    colormap_name: str,
    value_label: str,
    title_prefix: str,
    variable_key: str,
) -> dict[str, object]:
    return {
        "key": key,
        "label": label,
        "valueLabel": value_label,
        "images": [
            _encoded_webp_image(
                array,
                norm,
                colormap_name,
                value_label,
                f"{title_prefix} | {label} | Lead day {_lead_day_label(array, lead_day_index)}",
                variable_key,
            )
            for lead_day_index, array in enumerate(arrays)
        ],
    }


def _surface_comparison_multi_reference_depth_payload(
    fields: _ComputedMultiReferenceComparisonFields,
    variable_key: str,
    challenger_name: str,
    depth_key: str,
) -> dict[str, object]:
    challenger_fields = list(fields.challenger_fields)
    all_reference_fields = [
        reference_field for reference in fields.references for reference_field in reference.reference_fields
    ]
    all_error_fields = [
        challenger_field - reference_field
        for reference in fields.references
        for challenger_field, reference_field in zip(challenger_fields, reference.reference_fields, strict=True)
    ]
    all_absolute_error_fields = [abs(error_field) for error_field in all_error_fields]
    all_rmse_fields = [rmse_field for reference in fields.references for rmse_field in reference.rmse_fields]
    field_norm = _field_norm(variable_key, [*challenger_fields, *all_reference_fields])
    error_norm = _symmetric_norm(all_error_fields)
    absolute_error_norm = _positive_norm(all_absolute_error_fields)
    rmse_norm = _positive_norm(all_rmse_fields)
    variable_label = _variable_label_with_unit(variable_key)
    depth_label = _depth_label(challenger_fields[0])
    title_prefix = f"{variable_label} | {depth_label}"

    return {
        "key": depth_key,
        "label": depth_label,
        "leadDays": [
            _lead_day_label(challenger_field, lead_day_index)
            for lead_day_index, challenger_field in zip(
                fields.lead_day_indices,
                challenger_fields,
                strict=True,
            )
        ],
        "spatialStride": fields.spatial_stride,
        "challengerLayer": _encoded_images(
            "challenger",
            challenger_name,
            challenger_fields,
            field_norm,
            _field_colormap(variable_key),
            variable_label,
            title_prefix,
            variable_key,
        ),
        "references": [
            _surface_comparison_reference_payload(
                reference,
                challenger_fields=challenger_fields,
                challenger_name=challenger_name,
                variable_key=variable_key,
                field_norm=field_norm,
                error_norm=error_norm,
                absolute_error_norm=absolute_error_norm,
                rmse_norm=rmse_norm,
            )
            for reference in fields.references
        ],
    }


def _surface_comparison_reference_payload(
    reference: _ComputedReferenceComparisonFields,
    challenger_fields: Sequence[xarray.DataArray],
    challenger_name: str,
    variable_key: str,
    field_norm: Normalize,
    error_norm: Normalize,
    absolute_error_norm: Normalize,
    rmse_norm: Normalize,
) -> dict[str, object]:
    reference_fields = list(reference.reference_fields)
    error_fields = [
        challenger_field - reference_field
        for challenger_field, reference_field in zip(challenger_fields, reference_fields, strict=True)
    ]
    absolute_error_fields = [abs(error_field) for error_field in error_fields]
    rmse_fields = list(reference.rmse_fields)
    variable_label = _variable_label_with_unit(variable_key)
    title_prefix = f"{variable_label} | {_depth_label(reference_fields[0])}"
    return {
        "key": reference.key,
        "label": reference.label,
        "layers": [
            _encoded_images(
                "reference",
                reference.label,
                reference_fields,
                field_norm,
                _field_colormap(variable_key),
                variable_label,
                f"{reference.label} | {title_prefix}",
                variable_key,
            ),
            _encoded_images(
                "error",
                "Signed error",
                error_fields,
                error_norm,
                ERROR_COLORMAP,
                f"{variable_label} error",
                f"{challenger_name} vs {reference.label} | {title_prefix}",
                variable_key,
            ),
            _encoded_images(
                "absolute_error",
                "Absolute error",
                absolute_error_fields,
                absolute_error_norm,
                ABSOLUTE_ERROR_COLORMAP,
                f"{variable_label} absolute error",
                f"{challenger_name} vs {reference.label} | {title_prefix}",
                variable_key,
            ),
            _encoded_images(
                "rmse_over_dates",
                "RMSE over dates",
                rmse_fields,
                rmse_norm,
                RMSE_COLORMAP,
                f"{variable_label} RMSE",
                f"{challenger_name} vs {reference.label} | {title_prefix}",
                variable_key,
            ),
        ],
    }


def _surface_comparison_variable_payload(
    variable_key: str,
    depth_payloads: Sequence[dict[str, object]],
) -> dict[str, object]:
    return {
        "key": variable_key,
        "label": _variable_label(variable_key),
        "depths": list(depth_payloads),
    }


def _surface_comparison_payload(
    variable_payloads: Sequence[dict[str, object]],
    reference_name: str | None,
    title: str,
) -> dict[str, object]:
    return {
        "title": f"{title} against {reference_name}" if reference_name is not None else title,
        "variables": list(variable_payloads),
    }


def _surface_comparison_explorer_document(element_id: str, payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html, body {{
  margin: 0;
  padding: 0;
  background: transparent;
  color: #172033;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.ob-map-explorer {{
  box-sizing: border-box;
  width: 100%;
  min-width: 320px;
  padding: 14px 16px 12px;
  border: 1px solid #d8dee8;
  border-radius: 8px;
  background: #fbfcfe;
}}
.ob-map-header {{
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 10px;
}}
.ob-map-title {{
  font-size: 18px;
  font-weight: 650;
  line-height: 1.2;
}}
.ob-map-controls {{
  display: grid;
  gap: 9px;
  width: min(100%, 980px);
}}
.ob-map-control-row {{
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px 10px;
}}
.ob-map-chip-group {{
  display: inline-flex;
  max-width: 100%;
  overflow-x: auto;
  border: 1px solid #cbd5e1;
  border-radius: 7px;
  background: #ffffff;
}}
.ob-map-chip-button {{
  appearance: none;
  border: 0;
  border-right: 1px solid #cbd5e1;
  padding: 7px 10px;
  background: transparent;
  color: #334155;
  font: inherit;
  font-size: 13px;
  cursor: pointer;
  white-space: nowrap;
}}
.ob-map-chip-button:last-child {{
  border-right: 0;
}}
.ob-map-chip-button.active {{
  background: #0f5f8f;
  color: #ffffff;
}}
.ob-map-chip-button:disabled {{
  color: #64748b;
  cursor: default;
}}
.ob-map-depth-buttons.hidden {{
  visibility: hidden;
  pointer-events: none;
}}
.ob-map-lead-control {{
  display: grid;
  grid-template-columns: 11ch minmax(140px, 1fr);
  align-items: center;
  gap: 8px;
  color: #334155;
  font-size: 13px;
}}
.ob-map-lead-label {{
  display: block;
  text-align: right;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
  color: #334155;
  font-size: 13px;
}}
.ob-map-lead-control input {{
  accent-color: #0f5f8f;
  margin: 0;
}}
.ob-map-canvas-wrap {{
  position: relative;
  width: 100%;
  height: 560px;
  border: 1px solid #cfd8e3;
  border-radius: 6px;
  background: #ffffff;
  overflow: hidden;
}}
.ob-map-image {{
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #ffffff;
}}
.ob-map-status {{
  min-height: 18px;
  margin-top: 8px;
  color: #475569;
  font-size: 12px;
}}
@media (max-width: 760px) {{
  .ob-map-header {{
    display: block;
  }}
  .ob-map-controls {{
    margin-top: 10px;
    width: 100%;
  }}
  .ob-map-control-row,
  .ob-map-secondary-row {{
    justify-content: flex-start;
    width: 100%;
  }}
  .ob-map-canvas-wrap {{
    height: 420px;
  }}
}}
</style>
</head>
<body>
<div id="{element_id}" class="ob-map-explorer">
  <div class="ob-map-header">
    <div class="ob-map-title"></div>
    <div class="ob-map-controls">
      <div class="ob-map-control-row">
        <div class="ob-map-variable-buttons ob-map-chip-group"></div>
      </div>
      <div class="ob-map-control-row">
        <div class="ob-map-reference-buttons ob-map-chip-group"></div>
        <div class="ob-map-layer-buttons ob-map-chip-group"></div>
      </div>
      <div class="ob-map-control-row ob-map-secondary-row">
        <div class="ob-map-depth-buttons ob-map-chip-group"></div>
        <label class="ob-map-lead-control">
          <span class="ob-map-lead-label"></span>
          <input class="ob-map-lead-input" type="range">
        </label>
      </div>
    </div>
  </div>
  <div class="ob-map-canvas-wrap">
    <img class="ob-map-image" alt="Surface comparison map">
  </div>
  <div class="ob-map-status"></div>
</div>
<script>
(() => {{
  const payload = {payload_json};
  const root = document.getElementById("{element_id}");
  const title = root.querySelector(".ob-map-title");
  const variableButtons = root.querySelector(".ob-map-variable-buttons");
  const depthButtons = root.querySelector(".ob-map-depth-buttons");
  const referenceButtons = root.querySelector(".ob-map-reference-buttons");
  const layerButtons = root.querySelector(".ob-map-layer-buttons");
  const leadLabel = root.querySelector(".ob-map-lead-label");
  const leadInput = root.querySelector(".ob-map-lead-input");
  const image = root.querySelector(".ob-map-image");
  const status = root.querySelector(".ob-map-status");
  const variables = Object.fromEntries(payload.variables.map((variable) => [variable.key, variable]));
  let activeVariableKey = payload.variables[0].key;
  let activeDepthKey = payload.variables[0].depths[0].key;
  let activeReferenceKey = payload.variables[0].depths[0].references[0].key;
  let activeLayerKey = "error";
  let activeLeadIndex = 0;

  title.textContent = payload.title;

  leadInput.addEventListener("input", () => {{
    activeLeadIndex = Number(leadInput.value);
    render();
  }});

  updateControls();
  render();

  function currentVariable() {{
    return variables[activeVariableKey];
  }}

  function currentDepth() {{
    return Object.fromEntries(currentVariable().depths.map((depth) => [depth.key, depth]))[activeDepthKey];
  }}

  function currentReference() {{
    return Object.fromEntries(currentDepth().references.map((reference) => [reference.key, reference]))[
      activeReferenceKey
    ];
  }}

  function currentLayers() {{
    return [currentDepth().challengerLayer, ...currentReference().layers];
  }}

  function currentLayer() {{
    return Object.fromEntries(currentLayers().map((layer) => [layer.key, layer]))[activeLayerKey];
  }}

  function buildChipButtons(container, items, selectedKey, onSelect, disabled = false) {{
    container.replaceChildren();
    for (const item of items) {{
      const button = document.createElement("button");
      button.type = "button";
      button.className = "ob-map-chip-button";
      button.textContent = item.label;
      button.disabled = disabled;
      button.classList.toggle("active", item.key === selectedKey);
      button.addEventListener("click", () => {{
        if (!disabled) {{
          onSelect(item.key);
        }}
      }});
      container.appendChild(button);
    }}
  }}

  function updateControls() {{
    const variable = currentVariable();
    if (!variable.depths.some((depth) => depth.key === activeDepthKey)) {{
      activeDepthKey = variable.depths[0].key;
    }}
    const depth = currentDepth();
    if (!depth.references.some((reference) => reference.key === activeReferenceKey)) {{
      activeReferenceKey = depth.references[0].key;
    }}
    if (!currentLayers().some((layer) => layer.key === activeLayerKey)) {{
      activeLayerKey = currentLayers()[0].key;
    }}
    leadInput.min = 0;
    leadInput.max = depth.leadDays.length - 1;
    leadInput.step = 1;
    leadInput.value = activeLeadIndex;
    buildChipButtons(variableButtons, payload.variables, activeVariableKey, (variableKey) => {{
      activeVariableKey = variableKey;
      activeDepthKey = currentVariable().depths[0].key;
      activeReferenceKey = currentDepth().references[0].key;
      activeLeadIndex = Math.min(activeLeadIndex, currentDepth().leadDays.length - 1);
      updateControls();
      render();
    }});
    buildChipButtons(
      depthButtons,
      depthChipItems(variable.depths),
      activeDepthKey,
      (depthKey) => {{
        activeDepthKey = depthKey;
        activeReferenceKey = currentDepth().references[0].key;
        activeLeadIndex = Math.min(activeLeadIndex, currentDepth().leadDays.length - 1);
        updateControls();
        render();
      }},
      variable.depths.length <= 1,
    );
    depthButtons.classList.toggle("hidden", variable.depths.length <= 1);
    buildChipButtons(referenceButtons, depth.references, activeReferenceKey, (referenceKey) => {{
      activeReferenceKey = referenceKey;
      updateControls();
      render();
    }});
    buildChipButtons(layerButtons, currentLayers(), activeLayerKey, (layerKey) => {{
      activeLayerKey = layerKey;
      updateControls();
      render();
    }});
  }}

  function depthChipItems(depths) {{
    if (depths.length <= 1) {{
      return [{{ key: depths[0].key, label: "Surface" }}];
    }}
    return depths;
  }}

  function render() {{
    const variable = currentVariable();
    const depth = currentDepth();
    const layer = currentLayer();
    leadLabel.textContent = `Lead day ${{depth.leadDays[activeLeadIndex]}}`;
    image.src = layer.images[activeLeadIndex];
    image.alt = [
      payload.title,
      variable.label,
      layer.label,
      `lead day ${{depth.leadDays[activeLeadIndex]}}`,
    ].join(" - ");
    const depthText = variable.depths.length > 1 ? ` · ${{depth.label}}` : "";
    status.textContent = [
      `${{variable.label}}${{depthText}}`,
      currentReference().label,
      layer.label,
      `lead day ${{depth.leadDays[activeLeadIndex]}}`,
    ].join(" · ");
  }}
}})();
</script>
</body>
</html>"""


def _surface_comparison_iframe_html(document: str, height_pixels: int) -> str:
    escaped_document = html.escape(document, quote=True)
    return (
        f'<iframe srcdoc="{escaped_document}" '
        + 'style="width:100%; '
        + f'height:{height_pixels}px; border:0;" '
        + 'loading="lazy" sandbox="allow-scripts"></iframe>'
    )


def _html_without_iframe_warning(data: str) -> HTML:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Consider using IPython.display.IFrame instead",
            category=UserWarning,
        )
        return HTML(data)


def _json_ready_array(values: numpy.ndarray, decimals: int = 4) -> list:
    values = numpy.asarray(values, dtype=float)
    rounded_values = numpy.round(values, decimals=decimals).astype(object)
    rounded_values[~numpy.isfinite(values)] = None
    return rounded_values.tolist()


def _lagrangian_bounds(dataset: xarray.Dataset) -> dict[str, float]:
    standard_dataset = _standard_dataset(dataset)
    longitude = numpy.asarray(standard_dataset[Dimension.LONGITUDE.key()].values, dtype=float)
    latitude = numpy.asarray(standard_dataset[Dimension.LATITUDE.key()].values, dtype=float)
    longitude_minimum = float(numpy.nanmin(longitude))
    longitude_maximum = float(numpy.nanmax(longitude))
    latitude_minimum = float(numpy.nanmin(latitude))
    latitude_maximum = float(numpy.nanmax(latitude))
    longitude_padding = max(0.5, 0.02 * (longitude_maximum - longitude_minimum))
    latitude_padding = max(0.5, 0.02 * (latitude_maximum - latitude_minimum))
    return {
        "longitudeMinimum": longitude_minimum - longitude_padding,
        "longitudeMaximum": longitude_maximum + longitude_padding,
        "latitudeMinimum": latitude_minimum - latitude_padding,
        "latitudeMaximum": latitude_maximum + latitude_padding,
    }


def _particle_track_payload(particles: xarray.Dataset, time_count: int | None = None) -> dict[str, object]:
    selected_particles = particles if time_count is None else particles.isel({"time": slice(0, time_count)})
    return {
        "longitude": _json_ready_array(selected_particles["lon"].values),
        "latitude": _json_ready_array(selected_particles["lat"].values),
        "initialLongitude": _json_ready_array(selected_particles["lon0"].values),
        "initialLatitude": _json_ready_array(selected_particles["lat0"].values),
    }


def _trajectory_distances_kilometers(
    challenger_particles: xarray.Dataset,
    reference_particles: xarray.Dataset,
) -> numpy.ndarray:
    challenger_particles, reference_particles = xarray.align(challenger_particles, reference_particles, join="inner")
    latitude_reference_radians = numpy.deg2rad(reference_particles["lat"].values)
    latitude_distance = (challenger_particles["lat"].values - reference_particles["lat"].values) * 111.0
    longitude_delta = challenger_particles["lon"].values - reference_particles["lon"].values
    longitude_distance = longitude_delta * 111.0 * numpy.cos(latitude_reference_radians)
    return numpy.sqrt(latitude_distance**2 + longitude_distance**2)


def _surface_mask_field(dataset: xarray.Dataset, first_day_index: int) -> xarray.DataArray:
    field = _standard_dataset(dataset)[Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()]
    if Dimension.FIRST_DAY_DATETIME.key() in field.dims:
        field = field.isel({Dimension.FIRST_DAY_DATETIME.key(): first_day_index})
    if Dimension.LEAD_DAY_INDEX.key() in field.dims:
        field = field.isel({Dimension.LEAD_DAY_INDEX.key(): 0})
    elif Dimension.TIME.key() in field.dims:
        field = field.isel({Dimension.TIME.key(): 0})
    if Dimension.DEPTH.key() in field.dims:
        field = field.isel({Dimension.DEPTH.key(): 0})
    return field.compute().sortby([Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()])


def _lagrangian_land_mask_payload(dataset: xarray.Dataset, first_day_index: int) -> dict[str, object]:
    field = _surface_mask_field(dataset, first_day_index)
    thinned_field = _thin_field(field, _spatial_stride(field, DEFAULT_LAGRANGIAN_MAXIMUM_LAND_MASK_CELLS))
    land_mask = numpy.where(numpy.isfinite(thinned_field.values), 0, 1).astype(int)
    return {
        "longitude": _json_ready_array(thinned_field[Dimension.LONGITUDE.key()].values),
        "latitude": _json_ready_array(thinned_field[Dimension.LATITUDE.key()].values),
        "land": land_mask.tolist(),
    }


def _lagrangian_separation_scale_kilometers(
    challenger_particles: xarray.Dataset,
    reference_particles_by_key: Mapping[str, xarray.Dataset],
) -> float:
    distances = numpy.concatenate(
        [
            _trajectory_distances_kilometers(challenger_particles, reference_particles).ravel()
            for reference_particles in reference_particles_by_key.values()
        ]
    )
    finite_distances = distances[numpy.isfinite(distances)]
    if finite_distances.size == 0:
        return 1.0
    return max(1.0, float(numpy.nanpercentile(finite_distances, 95)))


def _reference_key(reference_name: str) -> str:
    reference_key = "".join(character.lower() if character.isalnum() else "_" for character in reference_name).strip(
        "_"
    )
    return reference_key or "reference"


def _lagrangian_payload(
    challenger_dataset: xarray.Dataset,
    reference_datasets: Mapping[str, xarray.Dataset],
    first_day_index: int,
    particle_count: int,
    seed: int,
    challenger_name: str,
    title: str,
) -> dict[str, object]:
    if not reference_datasets:
        raise ValueError("reference_datasets must contain at least one reference dataset.")
    if particle_count <= 0:
        raise ValueError("particle_count must be a positive integer.")
    challenger_standard_dataset = lagrangian_trajectory._harmonise_dataset(challenger_dataset)
    challenger_runs = lagrangian_trajectory._split_dataset(challenger_standard_dataset)
    if first_day_index < 0 or first_day_index >= len(challenger_runs):
        raise ValueError(f"first_day_index must be between 0 and {len(challenger_runs) - 1}.")
    initial_latitudes, initial_longitudes = lagrangian_trajectory._get_random_ocean_points_from_file(
        challenger_standard_dataset,
        variable_name=Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
        n=particle_count,
        seed=seed,
    )
    challenger_run_dataset = challenger_runs[first_day_index]
    challenger_particles = lagrangian_trajectory._get_particle_dataset(
        dataset=lagrangian_trajectory.surface_current_dataset(challenger_run_dataset),
        latitudes=initial_latitudes,
        longitudes=initial_longitudes,
    )
    reference_particles_by_key: dict[str, xarray.Dataset] = {}
    reference_labels_by_key: dict[str, str] = {}
    for reference_name, reference_dataset in reference_datasets.items():
        reference_standard_dataset = lagrangian_trajectory._harmonise_dataset(reference_dataset)
        reference_run_dataset = lagrangian_trajectory._split_dataset(reference_standard_dataset)[first_day_index]
        reference_particles = lagrangian_trajectory._get_particle_dataset(
            dataset=lagrangian_trajectory.surface_current_dataset(reference_run_dataset),
            latitudes=initial_latitudes,
            longitudes=initial_longitudes,
        )
        key = _reference_key(reference_name)
        reference_particles_by_key[key] = reference_particles
        reference_labels_by_key[key] = reference_name

    time_count = min(
        challenger_particles.sizes["time"],
        *(reference_particles.sizes["time"] for reference_particles in reference_particles_by_key.values()),
    )
    reference_payloads = [
        {
            "key": key,
            "label": reference_labels_by_key[key],
            "track": _particle_track_payload(reference_particles, time_count=time_count),
        }
        for key, reference_particles in reference_particles_by_key.items()
    ]
    return {
        "title": title,
        "challengerName": challenger_name,
        "particleCount": int(challenger_particles.sizes["particle"]),
        "timeLabels": [f"{index + 1:.1f}" for index in range(time_count)],
        "bounds": _lagrangian_bounds(challenger_dataset),
        "landMask": _lagrangian_land_mask_payload(challenger_standard_dataset, first_day_index),
        "separationScaleKilometers": _lagrangian_separation_scale_kilometers(
            challenger_particles,
            reference_particles_by_key,
        ),
        "challenger": _particle_track_payload(challenger_particles, time_count=time_count),
        "references": reference_payloads,
    }


def _lagrangian_explorer_document(element_id: str, payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html, body {{
  margin: 0;
  padding: 0;
  background: transparent;
  color: #172033;
  font-family: Arial, sans-serif;
}}
.ob-lagrangian {{
  border: 1px solid #cfd8e3;
  border-radius: 8px;
  background: #ffffff;
  overflow: hidden;
}}
.ob-lagrangian-header {{
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 16px;
  border-bottom: 1px solid #cfd8e3;
  background: #f8fafc;
}}
.ob-lagrangian-title {{
  font-size: 18px;
  font-weight: 650;
}}
.ob-lagrangian-subtitle {{
  margin-top: 4px;
  color: #64748b;
  font-size: 13px;
}}
.ob-lagrangian-controls {{
  display: grid;
  gap: 8px;
  min-width: 430px;
}}
.ob-lagrangian-control-row {{
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
  flex-wrap: wrap;
}}
.ob-lagrangian-chip-group {{
  display: inline-flex;
  overflow: hidden;
  border: 1px solid #cfd8e3;
  border-radius: 999px;
  background: #ffffff;
}}
.ob-lagrangian-chip {{
  border: 0;
  border-right: 1px solid #cfd8e3;
  padding: 7px 12px;
  background: transparent;
  color: #0f5f8f;
  font-size: 13px;
  cursor: pointer;
}}
.ob-lagrangian-chip:last-child {{
  border-right: 0;
}}
.ob-lagrangian-chip.active {{
  background: #0f5f8f;
  color: #ffffff;
}}
.ob-lagrangian-play {{
  width: 34px;
  height: 34px;
  border: 1px solid #cfd8e3;
  border-radius: 999px;
  background: #ffffff;
  color: #0f5f8f;
  cursor: pointer;
}}
.ob-lagrangian-zoom {{
  display: inline-flex;
  overflow: hidden;
  border: 1px solid #cfd8e3;
  border-radius: 999px;
  background: #ffffff;
}}
.ob-lagrangian-zoom button {{
  min-width: 34px;
  height: 34px;
  border: 0;
  border-right: 1px solid #cfd8e3;
  background: transparent;
  color: #0f5f8f;
  cursor: pointer;
  font-size: 13px;
}}
.ob-lagrangian-zoom button:last-child {{
  border-right: 0;
}}
.ob-lagrangian-slider {{
  display: grid;
  grid-template-columns: 13ch minmax(180px, 1fr);
  gap: 8px;
  align-items: center;
  min-width: 330px;
  color: #334155;
  font-size: 13px;
}}
.ob-lagrangian-slider span {{
  text-align: right;
  font-variant-numeric: tabular-nums;
}}
.ob-lagrangian-slider input {{
  accent-color: #0f5f8f;
}}
.ob-lagrangian-map {{
  height: 620px;
  background: linear-gradient(180deg, #eef7fb, #ffffff);
}}
.ob-lagrangian-map canvas {{
  display: block;
  width: 100%;
  height: 100%;
  cursor: grab;
  touch-action: none;
}}
.ob-lagrangian-map canvas.dragging {{
  cursor: grabbing;
}}
.ob-lagrangian-status {{
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 16px;
  border-top: 1px solid #cfd8e3;
  color: #64748b;
  font-size: 12px;
}}
.ob-lagrangian-legend {{
  display: flex;
  gap: 14px;
  align-items: center;
  flex-wrap: wrap;
}}
.ob-lagrangian-key {{
  display: inline-flex;
  gap: 6px;
  align-items: center;
}}
.ob-lagrangian-swatch {{
  width: 18px;
  height: 3px;
  border-radius: 999px;
  background: #000;
}}
.ob-lagrangian-swatch.ref {{ background: #64748b; }}
.ob-lagrangian-swatch.chal {{ background: #0f5f8f; }}
.ob-lagrangian-swatch.sep {{ background: linear-gradient(90deg, #0f9f8f, #f59e0b, #d1495b); }}
.ob-lagrangian-swatch.truth {{
  height: 8px;
  width: 8px;
  border: 1px solid #64748b;
  background: #ffffff;
}}
@media (max-width: 760px) {{
  .ob-lagrangian-header {{
    display: block;
  }}
  .ob-lagrangian-controls {{
    min-width: 0;
    margin-top: 12px;
  }}
  .ob-lagrangian-control-row {{
    justify-content: flex-start;
  }}
  .ob-lagrangian-map {{
    height: 440px;
  }}
}}
</style>
</head>
<body>
<div id="{element_id}" class="ob-lagrangian">
  <div class="ob-lagrangian-header">
    <div>
      <div class="ob-lagrangian-title"></div>
      <div class="ob-lagrangian-subtitle">Smooth visual interpolation between true daily particle positions.</div>
    </div>
    <div class="ob-lagrangian-controls">
      <div class="ob-lagrangian-control-row">
        <div class="ob-lagrangian-reference-buttons ob-lagrangian-chip-group"></div>
      </div>
      <div class="ob-lagrangian-control-row">
        <button class="ob-lagrangian-play" type="button">▶</button>
        <label class="ob-lagrangian-slider">
          <span class="ob-lagrangian-time-label"></span>
          <input class="ob-lagrangian-time-input" type="range" min="0" step="0.02" value="0">
        </label>
        <div class="ob-lagrangian-zoom" aria-label="Map zoom controls">
          <button class="ob-lagrangian-zoom-out" type="button" title="Zoom out">−</button>
          <button class="ob-lagrangian-zoom-in" type="button" title="Zoom in">+</button>
          <button class="ob-lagrangian-zoom-reset" type="button" title="Reset zoom">1:1</button>
        </div>
      </div>
    </div>
  </div>
  <div class="ob-lagrangian-map">
    <canvas width="1180" height="620"></canvas>
  </div>
  <div class="ob-lagrangian-status">
    <div class="ob-lagrangian-status-text"></div>
    <div class="ob-lagrangian-legend">
      <span class="ob-lagrangian-key"><span class="ob-lagrangian-swatch ref"></span>Reference trail</span>
      <span class="ob-lagrangian-key"><span class="ob-lagrangian-swatch chal"></span>Challenger trail</span>
      <span class="ob-lagrangian-key"><span class="ob-lagrangian-swatch sep"></span>Current separation distance</span>
      <span class="ob-lagrangian-key"><span class="ob-lagrangian-swatch truth"></span>Reached daily positions</span>
    </div>
  </div>
</div>
<script>
(() => {{
  const payload = {payload_json};
  const root = document.getElementById("{element_id}");
  const canvas = root.querySelector("canvas");
  const context = canvas.getContext("2d");
  const title = root.querySelector(".ob-lagrangian-title");
  const referenceButtons = root.querySelector(".ob-lagrangian-reference-buttons");
  const playButton = root.querySelector(".ob-lagrangian-play");
  const timeLabel = root.querySelector(".ob-lagrangian-time-label");
  const timeInput = root.querySelector(".ob-lagrangian-time-input");
  const zoomOutButton = root.querySelector(".ob-lagrangian-zoom-out");
  const zoomInButton = root.querySelector(".ob-lagrangian-zoom-in");
  const zoomResetButton = root.querySelector(".ob-lagrangian-zoom-reset");
  const statusText = root.querySelector(".ob-lagrangian-status-text");
  const references = Object.fromEntries(payload.references.map((reference) => [reference.key, reference]));
  const originalBounds = {{ ...payload.bounds }};
  let activeReferenceKey = payload.references[0].key;
  let activeTime = 0;
  let viewBounds = {{ ...originalBounds }};
  let dragStart = null;
  let animationFrame = null;
  let lastFrameTime = null;

  title.textContent = payload.title;
  timeInput.max = payload.timeLabels.length - 1;
  timeInput.addEventListener("input", () => {{
    activeTime = Number(timeInput.value);
    renderControls();
    draw();
  }});
  playButton.addEventListener("click", () => {{
    if (animationFrame !== null) {{
      stop();
      return;
    }}
    playButton.textContent = "Ⅱ";
    lastFrameTime = null;
    animationFrame = window.requestAnimationFrame(tick);
  }});
  zoomOutButton.addEventListener("click", () => {{
    zoom(0.5);
  }});
  zoomInButton.addEventListener("click", () => {{
    zoom(2.0);
  }});
  zoomResetButton.addEventListener("click", () => {{
    viewBounds = {{ ...originalBounds }};
    draw();
  }});
  canvas.addEventListener("pointerdown", (event) => {{
    if (event.button !== 0) return;
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add("dragging");
    dragStart = {{
      x: event.clientX,
      y: event.clientY,
      bounds: {{ ...viewBounds }},
    }};
  }});
  canvas.addEventListener("pointermove", (event) => {{
    if (dragStart === null) return;
    event.preventDefault();
    const longitudeSpan = dragStart.bounds.longitudeMaximum - dragStart.bounds.longitudeMinimum;
    const latitudeSpan = dragStart.bounds.latitudeMaximum - dragStart.bounds.latitudeMinimum;
    const longitudeDelta = ((event.clientX - dragStart.x) / canvas.clientWidth) * longitudeSpan;
    const latitudeDelta = ((event.clientY - dragStart.y) / canvas.clientHeight) * latitudeSpan;
    viewBounds = {{
      longitudeMinimum: dragStart.bounds.longitudeMinimum - longitudeDelta,
      longitudeMaximum: dragStart.bounds.longitudeMaximum - longitudeDelta,
      latitudeMinimum: dragStart.bounds.latitudeMinimum + latitudeDelta,
      latitudeMaximum: dragStart.bounds.latitudeMaximum + latitudeDelta,
    }};
    draw();
  }});
  canvas.addEventListener("pointerup", stopDrag);
  canvas.addEventListener("pointercancel", stopDrag);
  canvas.addEventListener("pointerleave", (event) => {{
    if (dragStart !== null && event.buttons === 0) stopDrag(event);
  }});

  renderControls();
  draw();

  function stop() {{
    window.cancelAnimationFrame(animationFrame);
    animationFrame = null;
    playButton.textContent = "▶";
  }}

  function tick(timestamp) {{
    if (lastFrameTime === null) {{
      lastFrameTime = timestamp;
    }}
    const elapsedSeconds = (timestamp - lastFrameTime) / 1000;
    lastFrameTime = timestamp;
    activeTime += elapsedSeconds * 0.85;
    if (activeTime > payload.timeLabels.length - 1) {{
      activeTime = 0;
    }}
    timeInput.value = activeTime;
    renderControls();
    draw();
    animationFrame = window.requestAnimationFrame(tick);
  }}

  function renderControls() {{
    referenceButtons.replaceChildren();
    for (const reference of payload.references) {{
      const button = document.createElement("button");
      button.type = "button";
      button.className = "ob-lagrangian-chip";
      button.textContent = reference.label;
      button.classList.toggle("active", reference.key === activeReferenceKey);
      button.addEventListener("click", () => {{
        activeReferenceKey = reference.key;
        renderControls();
        draw();
      }});
      referenceButtons.appendChild(button);
    }}
    timeLabel.textContent = `Lead day ${{(activeTime + 1).toFixed(1)}}`;
    statusText.textContent = [
      references[activeReferenceKey].label,
      `${{payload.particleCount}} sampled particles`,
      "interpolated display",
    ].join(" · ");
  }}

  function point(track, particleIndex, timeIndex) {{
    const longitude = track.longitude[particleIndex][timeIndex];
    const latitude = track.latitude[particleIndex][timeIndex];
    if (longitude === null || latitude === null) {{
      return null;
    }}
    return {{ longitude, latitude }};
  }}

  function interpolate(track, particleIndex, time) {{
    const lower = Math.max(0, Math.min(payload.timeLabels.length - 1, Math.floor(time)));
    const upper = Math.max(0, Math.min(payload.timeLabels.length - 1, lower + 1));
    const lowerPoint = point(track, particleIndex, lower);
    const upperPoint = point(track, particleIndex, upper);
    if (lowerPoint === null) {{
      return upperPoint;
    }}
    if (upperPoint === null || upper === lower) {{
      return lowerPoint;
    }}
    const fraction = time - lower;
    return {{
      longitude: lowerPoint.longitude * (1 - fraction) + upperPoint.longitude * fraction,
      latitude: lowerPoint.latitude * (1 - fraction) + upperPoint.latitude * fraction,
    }};
  }}

  function project(point) {{
    const bounds = viewBounds;
    const x = (
      (point.longitude - bounds.longitudeMinimum) / (bounds.longitudeMaximum - bounds.longitudeMinimum)
    ) * canvas.width;
    const y = canvas.height - (
      (point.latitude - bounds.latitudeMinimum) / (bounds.latitudeMaximum - bounds.latitudeMinimum)
    ) * canvas.height;
    return {{ x, y }};
  }}

  function zoom(factor) {{
    zoomAt(
      factor,
      (viewBounds.longitudeMinimum + viewBounds.longitudeMaximum) / 2,
      (viewBounds.latitudeMinimum + viewBounds.latitudeMaximum) / 2,
    );
  }}

  function zoomAt(factor, longitudeCenter, latitudeCenter) {{
    const longitudeHalfSpan = constrainedSpan(
      viewBounds.longitudeMaximum - viewBounds.longitudeMinimum,
      originalBounds.longitudeMaximum - originalBounds.longitudeMinimum,
      factor,
    ) / 2;
    const latitudeHalfSpan = constrainedSpan(
      viewBounds.latitudeMaximum - viewBounds.latitudeMinimum,
      originalBounds.latitudeMaximum - originalBounds.latitudeMinimum,
      factor,
    ) / 2;
    viewBounds = {{
      longitudeMinimum: longitudeCenter - longitudeHalfSpan,
      longitudeMaximum: longitudeCenter + longitudeHalfSpan,
      latitudeMinimum: latitudeCenter - latitudeHalfSpan,
      latitudeMaximum: latitudeCenter + latitudeHalfSpan,
    }};
    draw();
  }}

  function constrainedSpan(currentSpan, originalSpan, factor) {{
    const minimumSpan = originalSpan / 80;
    const maximumSpan = originalSpan * 2;
    return Math.max(minimumSpan, Math.min(maximumSpan, currentSpan / factor));
  }}

  function canvasPoint(event) {{
    const rectangle = canvas.getBoundingClientRect();
    return {{
      x: event.clientX - rectangle.left,
      y: event.clientY - rectangle.top,
    }};
  }}

  function stopDrag(event) {{
    if (dragStart === null) return;
    if (event && canvas.hasPointerCapture(event.pointerId)) {{
      canvas.releasePointerCapture(event.pointerId);
    }}
    canvas.classList.remove("dragging");
    dragStart = null;
  }}

  function draw() {{
    context.clearRect(0, 0, canvas.width, canvas.height);
    drawBackground();
    const reference = references[activeReferenceKey];
    drawDailyMarkers(payload.challenger, reference.track, activeTime);
    for (let particleIndex = 0; particleIndex < payload.particleCount; particleIndex += 1) {{
      const referencePoint = interpolate(reference.track, particleIndex, activeTime);
      const challengerPoint = interpolate(payload.challenger, particleIndex, activeTime);
      if (referencePoint === null || challengerPoint === null) {{
        continue;
      }}
      const separationColorValue = separationColor(referencePoint, challengerPoint);
      drawTrail(reference.track, particleIndex, referencePoint, activeTime, "rgba(71, 85, 105, 0.24)", 1.1);
      drawTrail(payload.challenger, particleIndex, challengerPoint, activeTime, "rgba(15, 95, 143, 0.42)", 1.5);
      drawSeparation(referencePoint, challengerPoint, separationColorValue);
      drawPosition(referencePoint, "#64748b", 2.0, "#ffffff");
      drawPosition(challengerPoint, "#0f5f8f", 3.0, "#ffffff");
    }}
    drawFrame();
  }}

  function drawBackground() {{
    context.fillStyle = "#eef7fb";
    context.fillRect(0, 0, canvas.width, canvas.height);
    drawLandMask();
    context.strokeStyle = "rgba(51, 65, 85, 0.18)";
    context.lineWidth = 1;
    const bounds = viewBounds;
    const longitudeStep = niceStep(bounds.longitudeMaximum - bounds.longitudeMinimum);
    const latitudeStep = niceStep(bounds.latitudeMaximum - bounds.latitudeMinimum);
    for (
      let longitude = Math.ceil(bounds.longitudeMinimum / longitudeStep) * longitudeStep;
      longitude <= bounds.longitudeMaximum;
      longitude += longitudeStep
    ) {{
      line(
        project({{ longitude, latitude: bounds.latitudeMinimum }}),
        project({{ longitude, latitude: bounds.latitudeMaximum }}),
      );
    }}
    for (
      let latitude = Math.ceil(bounds.latitudeMinimum / latitudeStep) * latitudeStep;
      latitude <= bounds.latitudeMaximum;
      latitude += latitudeStep
    ) {{
      line(
        project({{ longitude: bounds.longitudeMinimum, latitude }}),
        project({{ longitude: bounds.longitudeMaximum, latitude }}),
      );
    }}
  }}

  function niceStep(span) {{
    if (span > 180) return 60;
    if (span > 90) return 30;
    if (span > 35) return 10;
    if (span > 12) return 5;
    return 2;
  }}

  function drawLandMask() {{
    const landMask = payload.landMask;
    if (!landMask || landMask.land.length === 0) return;
    const longitudes = landMask.longitude;
    const latitudes = landMask.latitude;
    const longitudeStep = coordinateStep(longitudes);
    const latitudeStep = coordinateStep(latitudes);
    context.fillStyle = "{LAND_BACKGROUND_COLOR}";
    for (let latitudeIndex = 0; latitudeIndex < latitudes.length; latitudeIndex += 1) {{
      for (let longitudeIndex = 0; longitudeIndex < longitudes.length; longitudeIndex += 1) {{
        if (landMask.land[latitudeIndex][longitudeIndex] !== 1) continue;
        const west = longitudes[longitudeIndex] - longitudeStep / 2;
        const east = longitudes[longitudeIndex] + longitudeStep / 2;
        const south = latitudes[latitudeIndex] - latitudeStep / 2;
        const north = latitudes[latitudeIndex] + latitudeStep / 2;
        const northwest = project({{ longitude: west, latitude: north }});
        const southeast = project({{ longitude: east, latitude: south }});
        context.fillRect(northwest.x, northwest.y, southeast.x - northwest.x + 1, southeast.y - northwest.y + 1);
      }}
    }}
  }}

  function coordinateStep(values) {{
    if (values.length < 2) return 1;
    return Math.abs(values[1] - values[0]);
  }}

  function drawDailyMarkers(challengerTrack, referenceTrack, time) {{
    context.lineWidth = 0.8;
    const lastReachedIndex = Math.floor(time);
    for (let particleIndex = 0; particleIndex < payload.particleCount; particleIndex += 1) {{
      for (let timeIndex = 0; timeIndex <= lastReachedIndex; timeIndex += 1) {{
        const referencePoint = point(referenceTrack, particleIndex, timeIndex);
        const challengerPoint = point(challengerTrack, particleIndex, timeIndex);
        context.fillStyle = "rgba(255, 255, 255, 0.62)";
        if (referencePoint !== null) {{
          context.strokeStyle = "rgba(100, 116, 139, 0.42)";
          circle(project(referencePoint), 1.7, true);
        }}
        if (challengerPoint !== null) {{
          context.strokeStyle = "rgba(15, 95, 143, 0.42)";
          circle(project(challengerPoint), 1.7, true);
        }}
      }}
    }}
  }}

  function drawTrail(track, particleIndex, currentPoint, time, color, width) {{
    const lower = Math.floor(time);
    context.strokeStyle = color;
    context.lineWidth = width;
    let previousProjectedPoint = null;
    context.beginPath();
    for (let timeIndex = 0; timeIndex <= lower; timeIndex += 1) {{
      const trackPoint = point(track, particleIndex, timeIndex);
      if (trackPoint === null) {{
        previousProjectedPoint = null;
        continue;
      }}
      const projectedPoint = project(trackPoint);
      if (previousProjectedPoint === null || tooFarApart(previousProjectedPoint, projectedPoint)) {{
        context.moveTo(projectedPoint.x, projectedPoint.y);
      }} else {{
        context.lineTo(projectedPoint.x, projectedPoint.y);
      }}
      previousProjectedPoint = projectedPoint;
    }}
    const currentProjectedPoint = project(currentPoint);
    if (previousProjectedPoint === null || tooFarApart(previousProjectedPoint, currentProjectedPoint)) {{
      context.moveTo(currentProjectedPoint.x, currentProjectedPoint.y);
    }} else {{
      context.lineTo(currentProjectedPoint.x, currentProjectedPoint.y);
    }}
    context.stroke();
  }}

  function drawSeparation(referencePoint, challengerPoint, color) {{
    const referenceProjectedPoint = project(referencePoint);
    const challengerProjectedPoint = project(challengerPoint);
    if (tooFarApart(referenceProjectedPoint, challengerProjectedPoint)) {{
      return;
    }}
    context.strokeStyle = color.replace("1)", "0.68)");
    context.lineWidth = 1.8;
    line(referenceProjectedPoint, challengerProjectedPoint);
  }}

  function drawPosition(pointToDraw, fill, radius, stroke) {{
    const projectedPoint = project(pointToDraw);
    context.fillStyle = fill;
    context.strokeStyle = stroke;
    context.lineWidth = 1.2;
    circle(projectedPoint, radius, true);
  }}

  function circle(projectedPoint, radius, withStroke) {{
    context.beginPath();
    context.arc(projectedPoint.x, projectedPoint.y, radius, 0, Math.PI * 2);
    context.fill();
    if (withStroke) {{
      context.stroke();
    }}
  }}

  function drawFrame() {{
    context.strokeStyle = "#cfd8e3";
    context.lineWidth = 2;
    context.strokeRect(1, 1, canvas.width - 2, canvas.height - 2);
  }}

  function line(a, b) {{
    context.beginPath();
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.stroke();
  }}

  function tooFarApart(a, b) {{
    return Math.abs(a.x - b.x) > canvas.width * 0.5;
  }}

  function separationColor(referencePoint, challengerPoint) {{
    const distance = separationDistanceKilometers(referencePoint, challengerPoint);
    const ratio = Math.max(0, Math.min(1, distance / payload.separationScaleKilometers));
    if (ratio < 0.5) {{
      const fraction = ratio / 0.5;
      return colorMix([15, 159, 143], [245, 158, 11], fraction);
    }}
    return colorMix([245, 158, 11], [209, 73, 91], (ratio - 0.5) / 0.5);
  }}

  function separationDistanceKilometers(referencePoint, challengerPoint) {{
    const latitudeRadians = referencePoint.latitude * Math.PI / 180;
    const latitudeDistance = (challengerPoint.latitude - referencePoint.latitude) * 111;
    const longitudeDistance = (challengerPoint.longitude - referencePoint.longitude) * 111 * Math.cos(latitudeRadians);
    return Math.sqrt(latitudeDistance * latitudeDistance + longitudeDistance * longitudeDistance);
  }}

  function colorMix(start, end, fraction) {{
    const values = start.map((value, index) => Math.round(value * (1 - fraction) + end[index] * fraction));
    return `rgba(${{values[0]}}, ${{values[1]}}, ${{values[2]}}, 1)`;
  }}
}})();
</script>
</body>
</html>"""


def _lagrangian_explorer_iframe_html(document: str, height_pixels: int) -> str:
    escaped_document = html.escape(document, quote=True)
    return (
        f'<iframe srcdoc="{escaped_document}" '
        + 'style="width:100%; '
        + f'height:{height_pixels}px; border:0;" '
        + 'loading="lazy" sandbox="allow-scripts"></iframe>'
    )


def _decimated_contour_values(values: object, maximum_points: int) -> list:
    array = numpy.asarray(values, dtype=float)
    if array.size == 0:
        return []
    stride = max(1, int(numpy.ceil(array.size / maximum_points)))
    return _json_ready_array(array[::stride], decimals=4)


def _eddy_record(row: pandas.Series, contours: pandas.DataFrame, maximum_contour_points: int) -> dict[str, object]:
    contour = contours.loc[contours["detection_index"] == row.name]
    contour_latitudes: list = []
    contour_longitudes: list = []
    if not contour.empty:
        contour_row = contour.iloc[0]
        contour_latitudes = _decimated_contour_values(
            contour_row[eddies.CONTOUR_LATITUDES_COLUMN],
            maximum_contour_points,
        )
        contour_longitudes = _decimated_contour_values(
            contour_row[eddies.CONTOUR_LONGITUDES_COLUMN],
            maximum_contour_points,
        )
    return {
        "id": int(row.name),
        "latitude": round(float(row[eddies.LATITUDE_COLUMN]), 4),
        "longitude": round(float(row[eddies.LONGITUDE_COLUMN]), 4),
        "polarity": str(row[eddies.POLARITY_COLUMN]),
        "contourLatitude": contour_latitudes,
        "contourLongitude": contour_longitudes,
    }


def _eddy_frames_payload(
    challenger_detections: pandas.DataFrame,
    challenger_contours: pandas.DataFrame,
    reference_detections: pandas.DataFrame,
    reference_contours: pandas.DataFrame,
    matches: pandas.DataFrame,
    lead_day_indices: Sequence[int],
    maximum_contour_points: int,
) -> list[dict[str, object]]:
    frames = []
    for lead_day_index in lead_day_indices:
        challenger_subset = challenger_detections.loc[challenger_detections[eddies.LEAD_DAY_COLUMN] == lead_day_index]
        reference_subset = reference_detections.loc[reference_detections[eddies.LEAD_DAY_COLUMN] == lead_day_index]
        match_subset = matches.loc[matches[eddies.LEAD_DAY_COLUMN] == lead_day_index] if not matches.empty else matches
        matched_challenger_indices = set(match_subset.get("challenger_detection_index", pandas.Series(dtype=int)))
        matched_reference_indices = set(match_subset.get("reference_detection_index", pandas.Series(dtype=int)))
        challenger_records = {
            int(row_index): _eddy_record(row, challenger_contours, maximum_contour_points)
            for row_index, row in challenger_subset.iterrows()
        }
        reference_records = {
            int(row_index): _eddy_record(row, reference_contours, maximum_contour_points)
            for row_index, row in reference_subset.iterrows()
        }
        frame_matches = []
        for _, match_row in match_subset.iterrows():
            challenger_index = int(match_row["challenger_detection_index"])
            reference_index = int(match_row["reference_detection_index"])
            if challenger_index not in challenger_records or reference_index not in reference_records:
                continue
            frame_matches.append(
                {
                    "challenger": challenger_records[challenger_index],
                    "reference": reference_records[reference_index],
                    "distanceKilometers": round(float(match_row[eddies.DISTANCE_COLUMN]), 2),
                }
            )
        frames.append(
            {
                "leadDay": int(lead_day_index) + 1,
                "matches": frame_matches,
                "spurious": [
                    record
                    for record_index, record in challenger_records.items()
                    if record_index not in matched_challenger_indices
                ],
                "missed": [
                    record
                    for record_index, record in reference_records.items()
                    if record_index not in matched_reference_indices
                ],
            }
        )
    return frames


def _eddy_reference_payload(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    reference_name: str,
    first_day_index: int,
    lead_day_indices: Sequence[int],
    maximum_contour_points: int,
    detection_parameters: Mapping[str, float | int],
) -> dict[str, object]:
    challenger_detections = eddies.detect_mesoscale_eddies(
        challenger_dataset,
        first_day_index=first_day_index,
        lead_day_indices=list(lead_day_indices),
        background_sigma_grid=detection_parameters["background_sigma_grid"],
        detection_sigma_grid=detection_parameters["detection_sigma_grid"],
        min_distance_grid=detection_parameters["min_distance_grid"],
        amplitude_threshold_meters=detection_parameters["amplitude_threshold_meters"],
        max_abs_latitude_degrees=detection_parameters["max_abs_latitude_degrees"],
    )
    challenger_contours = eddies.mesoscale_eddy_contours_from_detections(
        challenger_detections,
        challenger_dataset,
        first_day_index=first_day_index,
        background_sigma_grid=detection_parameters["background_sigma_grid"],
        detection_sigma_grid=detection_parameters["detection_sigma_grid"],
        amplitude_threshold_meters=detection_parameters["amplitude_threshold_meters"],
        max_abs_latitude_degrees=detection_parameters["max_abs_latitude_degrees"],
        contour_level_step_meters=detection_parameters["contour_level_step_meters"],
        min_contour_pixel_count=detection_parameters["min_contour_pixel_count"],
        max_contour_pixel_count=detection_parameters["max_contour_pixel_count"],
        min_contour_convexity=detection_parameters["min_contour_convexity"],
    )
    challenger_detections = eddies.filter_mesoscale_eddy_detections_by_contours(
        challenger_detections,
        challenger_contours,
    )
    reference_detections = eddies.detect_mesoscale_eddies(
        reference_dataset,
        first_day_index=first_day_index,
        lead_day_indices=list(lead_day_indices),
        background_sigma_grid=detection_parameters["background_sigma_grid"],
        detection_sigma_grid=detection_parameters["detection_sigma_grid"],
        min_distance_grid=detection_parameters["min_distance_grid"],
        amplitude_threshold_meters=detection_parameters["amplitude_threshold_meters"],
        max_abs_latitude_degrees=detection_parameters["max_abs_latitude_degrees"],
    )
    reference_contours = eddies.mesoscale_eddy_contours_from_detections(
        reference_detections,
        reference_dataset,
        first_day_index=first_day_index,
        background_sigma_grid=detection_parameters["background_sigma_grid"],
        detection_sigma_grid=detection_parameters["detection_sigma_grid"],
        amplitude_threshold_meters=detection_parameters["amplitude_threshold_meters"],
        max_abs_latitude_degrees=detection_parameters["max_abs_latitude_degrees"],
        contour_level_step_meters=detection_parameters["contour_level_step_meters"],
        min_contour_pixel_count=detection_parameters["min_contour_pixel_count"],
        max_contour_pixel_count=detection_parameters["max_contour_pixel_count"],
        min_contour_convexity=detection_parameters["min_contour_convexity"],
    )
    reference_detections = eddies.filter_mesoscale_eddy_detections_by_contours(
        reference_detections,
        reference_contours,
    )
    matches = eddies.match_mesoscale_eddies(
        challenger_detections,
        reference_detections,
        max_match_distance_km=detection_parameters["max_match_distance_km"],
    )
    return {
        "key": _reference_key(reference_name),
        "label": reference_name,
        "frames": _eddy_frames_payload(
            challenger_detections,
            challenger_contours,
            reference_detections,
            reference_contours,
            matches,
            lead_day_indices,
            maximum_contour_points,
        ),
    }


def _eddy_payload(
    challenger_dataset: xarray.Dataset,
    reference_datasets: Mapping[str, xarray.Dataset],
    first_day_index: int,
    lead_day_indices: Sequence[int] | None,
    maximum_contour_points: int,
    title: str,
    detection_parameters: Mapping[str, float | int],
) -> dict[str, object]:
    if not reference_datasets:
        raise ValueError("reference_datasets must contain at least one reference dataset.")
    resolved_lead_day_indices = (
        tuple(eddies._lead_day_indices(challenger_dataset))
        if lead_day_indices is None
        else tuple(index for index in lead_day_indices if index in eddies._lead_day_indices(challenger_dataset))
    )
    if not resolved_lead_day_indices:
        raise ValueError("lead_day_indices must contain at least one available lead day.")
    return {
        "title": title,
        "bounds": _lagrangian_bounds(challenger_dataset),
        "landMask": _lagrangian_land_mask_payload(challenger_dataset, first_day_index),
        "references": [
            _eddy_reference_payload(
                challenger_dataset,
                reference_dataset,
                reference_name,
                first_day_index,
                resolved_lead_day_indices,
                maximum_contour_points,
                detection_parameters,
            )
            for reference_name, reference_dataset in reference_datasets.items()
        ],
    }


def _eddy_explorer_document(element_id: str, payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html, body {{
  margin: 0;
  padding: 0;
  background: transparent;
  color: #172033;
  font-family: Arial, sans-serif;
}}
.ob-eddy {{
  border: 1px solid #cfd8e3;
  border-radius: 8px;
  background: #ffffff;
  overflow: hidden;
}}
.ob-eddy-header {{
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 16px;
  border-bottom: 1px solid #cfd8e3;
  background: #f8fafc;
}}
.ob-eddy-title {{
  font-size: 18px;
  font-weight: 650;
}}
.ob-eddy-subtitle {{
  margin-top: 4px;
  color: #64748b;
  font-size: 13px;
}}
.ob-eddy-controls {{
  display: grid;
  gap: 8px;
  min-width: 440px;
}}
.ob-eddy-row {{
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
  flex-wrap: wrap;
}}
.ob-eddy-chip-group {{
  display: inline-flex;
  overflow: hidden;
  border: 1px solid #cfd8e3;
  border-radius: 999px;
  background: #ffffff;
}}
.ob-eddy-chip {{
  border: 0;
  border-right: 1px solid #cfd8e3;
  padding: 7px 12px;
  background: transparent;
  color: #0f5f8f;
  font-size: 13px;
  cursor: pointer;
}}
.ob-eddy-chip:last-child {{
  border-right: 0;
}}
.ob-eddy-chip.active {{
  background: #0f5f8f;
  color: #ffffff;
}}
.ob-eddy-play {{
  width: 34px;
  height: 34px;
  border: 1px solid #cfd8e3;
  border-radius: 999px;
  background: #ffffff;
  color: #0f5f8f;
  cursor: pointer;
}}
.ob-eddy-slider {{
  display: grid;
  grid-template-columns: 11ch minmax(180px, 1fr);
  gap: 8px;
  align-items: center;
  min-width: 315px;
  color: #334155;
  font-size: 13px;
}}
.ob-eddy-slider span {{
  text-align: right;
  font-variant-numeric: tabular-nums;
}}
.ob-eddy-slider input {{
  accent-color: #0f5f8f;
}}
.ob-eddy-map {{
  height: 620px;
  background: linear-gradient(180deg, #eef7fb, #ffffff);
}}
.ob-eddy-map canvas {{
  display: block;
  width: 100%;
  height: 100%;
}}
.ob-eddy-status {{
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 16px;
  border-top: 1px solid #cfd8e3;
  color: #64748b;
  font-size: 12px;
}}
.ob-eddy-legend {{
  display: flex;
  gap: 14px;
  align-items: center;
  flex-wrap: wrap;
}}
.ob-eddy-key {{
  display: inline-flex;
  gap: 6px;
  align-items: center;
}}
.ob-eddy-swatch {{
  width: 18px;
  height: 3px;
  border-radius: 999px;
}}
.ob-eddy-swatch.match {{ background: #0f9f8f; }}
.ob-eddy-swatch.spurious {{ background: #d1495b; }}
.ob-eddy-swatch.missed {{ background: #f59e0b; }}
.ob-eddy-swatch.line {{ background: #475569; }}
</style>
</head>
<body>
<div id="{element_id}" class="ob-eddy">
  <div class="ob-eddy-header">
    <div>
      <div class="ob-eddy-title"></div>
      <div class="ob-eddy-subtitle">Discrete SSH eddy detections per lead day; contours are not interpolated.</div>
    </div>
    <div class="ob-eddy-controls">
      <div class="ob-eddy-row">
        <div class="ob-eddy-reference-buttons ob-eddy-chip-group"></div>
      </div>
      <div class="ob-eddy-row">
        <button class="ob-eddy-play" type="button">▶</button>
        <label class="ob-eddy-slider">
          <span class="ob-eddy-lead-label"></span>
          <input class="ob-eddy-lead-input" type="range" min="0" step="1" value="0">
        </label>
      </div>
    </div>
  </div>
  <div class="ob-eddy-map"><canvas width="1180" height="620"></canvas></div>
  <div class="ob-eddy-status">
    <div class="ob-eddy-status-text"></div>
    <div class="ob-eddy-legend">
      <span class="ob-eddy-key"><span class="ob-eddy-swatch match"></span>Matched eddy</span>
      <span class="ob-eddy-key"><span class="ob-eddy-swatch spurious"></span>Spurious challenger</span>
      <span class="ob-eddy-key"><span class="ob-eddy-swatch missed"></span>Missed reference</span>
      <span class="ob-eddy-key"><span class="ob-eddy-swatch line"></span>Match displacement</span>
    </div>
  </div>
</div>
<script>
(() => {{
  const payload = {payload_json};
  const root = document.getElementById("{element_id}");
  const canvas = root.querySelector("canvas");
  const context = canvas.getContext("2d");
  const title = root.querySelector(".ob-eddy-title");
  const referenceButtons = root.querySelector(".ob-eddy-reference-buttons");
  const playButton = root.querySelector(".ob-eddy-play");
  const leadInput = root.querySelector(".ob-eddy-lead-input");
  const leadLabel = root.querySelector(".ob-eddy-lead-label");
  const statusText = root.querySelector(".ob-eddy-status-text");
  const references = Object.fromEntries(payload.references.map((reference) => [reference.key, reference]));
  let activeReferenceKey = payload.references[0].key;
  let activeLeadIndex = 0;
  let timer = null;

  title.textContent = payload.title;
  leadInput.addEventListener("input", () => {{
    activeLeadIndex = Number(leadInput.value);
    renderControls();
    draw();
  }});
  playButton.addEventListener("click", () => {{
    if (timer !== null) {{
      window.clearInterval(timer);
      timer = null;
      playButton.textContent = "▶";
      return;
    }}
    playButton.textContent = "Ⅱ";
    timer = window.setInterval(() => {{
      activeLeadIndex = (activeLeadIndex + 1) % references[activeReferenceKey].frames.length;
      leadInput.value = activeLeadIndex;
      renderControls();
      draw();
    }}, 850);
  }});

  renderControls();
  draw();

  function renderControls() {{
    referenceButtons.replaceChildren();
    for (const reference of payload.references) {{
      const button = document.createElement("button");
      button.type = "button";
      button.className = "ob-eddy-chip";
      button.textContent = reference.label;
      button.classList.toggle("active", reference.key === activeReferenceKey);
      button.addEventListener("click", () => {{
        activeReferenceKey = reference.key;
        activeLeadIndex = Math.min(activeLeadIndex, references[activeReferenceKey].frames.length - 1);
        leadInput.value = activeLeadIndex;
        renderControls();
        draw();
      }});
      referenceButtons.appendChild(button);
    }}
    const frame = references[activeReferenceKey].frames[activeLeadIndex];
    leadInput.max = references[activeReferenceKey].frames.length - 1;
    leadLabel.textContent = `Lead day ${{frame.leadDay}}`;
    statusText.textContent = [
      references[activeReferenceKey].label,
      `lead day ${{frame.leadDay}}`,
      `${{frame.matches.length}} matches`,
      `${{frame.spurious.length}} spurious`,
      `${{frame.missed.length}} missed`,
    ].join(" · ");
  }}

  function project(point) {{
    const bounds = payload.bounds;
    const longitudeSpan = bounds.longitudeMaximum - bounds.longitudeMinimum;
    const latitudeSpan = bounds.latitudeMaximum - bounds.latitudeMinimum;
    return {{
      x: ((point.longitude - bounds.longitudeMinimum) / longitudeSpan) * canvas.width,
      y: canvas.height - ((point.latitude - bounds.latitudeMinimum) / latitudeSpan) * canvas.height,
    }};
  }}

  function draw() {{
    context.clearRect(0, 0, canvas.width, canvas.height);
    drawBackground();
    const frame = references[activeReferenceKey].frames[activeLeadIndex];
    for (const match of frame.matches) drawMatchLine(match.reference, match.challenger);
    for (const missed of frame.missed) drawEddy(missed, "rgba(245, 158, 11, 0.15)", "rgba(245, 158, 11, 0.88)", 2.0);
    for (const spurious of frame.spurious) drawEddy(spurious, "rgba(209, 73, 91, 0.13)", "rgba(209, 73, 91, 0.9)", 2.0);
    for (const match of frame.matches) {{
      drawEddy(match.reference, "rgba(100, 116, 139, 0.10)", "rgba(100, 116, 139, 0.65)", 1.3);
      drawEddy(match.challenger, "rgba(15, 159, 143, 0.16)", "rgba(15, 159, 143, 0.95)", 2.1);
    }}
    drawFrame();
  }}

  function drawBackground() {{
    context.fillStyle = "#eef7fb";
    context.fillRect(0, 0, canvas.width, canvas.height);
    drawLandMask();
    context.strokeStyle = "rgba(51, 65, 85, 0.18)";
    context.lineWidth = 1;
    const bounds = payload.bounds;
    const firstLongitude = Math.ceil(bounds.longitudeMinimum / 20) * 20;
    const firstLatitude = Math.ceil(bounds.latitudeMinimum / 10) * 10;
    for (let longitude = firstLongitude; longitude <= bounds.longitudeMaximum; longitude += 20) {{
      const southPoint = project({{ longitude, latitude: bounds.latitudeMinimum }});
      const northPoint = project({{ longitude, latitude: bounds.latitudeMaximum }});
      line(southPoint, northPoint);
    }}
    for (let latitude = firstLatitude; latitude <= bounds.latitudeMaximum; latitude += 10) {{
      const westPoint = project({{ longitude: bounds.longitudeMinimum, latitude }});
      const eastPoint = project({{ longitude: bounds.longitudeMaximum, latitude }});
      line(westPoint, eastPoint);
    }}
  }}

  function drawLandMask() {{
    const landMask = payload.landMask;
    if (!landMask || landMask.land.length === 0) return;
    const longitudes = landMask.longitude;
    const latitudes = landMask.latitude;
    const longitudeStep = coordinateStep(longitudes);
    const latitudeStep = coordinateStep(latitudes);
    context.fillStyle = "{LAND_BACKGROUND_COLOR}";
    for (let latitudeIndex = 0; latitudeIndex < latitudes.length; latitudeIndex += 1) {{
      for (let longitudeIndex = 0; longitudeIndex < longitudes.length; longitudeIndex += 1) {{
        if (landMask.land[latitudeIndex][longitudeIndex] !== 1) continue;
        const west = longitudes[longitudeIndex] - longitudeStep / 2;
        const east = longitudes[longitudeIndex] + longitudeStep / 2;
        const south = latitudes[latitudeIndex] - latitudeStep / 2;
        const north = latitudes[latitudeIndex] + latitudeStep / 2;
        const northwest = project({{ longitude: west, latitude: north }});
        const southeast = project({{ longitude: east, latitude: south }});
        context.fillRect(northwest.x, northwest.y, southeast.x - northwest.x + 1, southeast.y - northwest.y + 1);
      }}
    }}
  }}

  function coordinateStep(values) {{
    if (values.length < 2) return 1;
    return Math.abs(values[1] - values[0]);
  }}

  function drawMatchLine(reference, challenger) {{
    context.strokeStyle = "rgba(71, 85, 105, 0.44)";
    context.lineWidth = 1.0;
    line(project(reference), project(challenger));
  }}

  function drawEddy(candidate, fill, stroke, width) {{
    const center = project(candidate);
    context.fillStyle = fill;
    context.strokeStyle = stroke;
    context.lineWidth = width;
    if (candidate.contourLongitude.length >= 3) {{
      context.beginPath();
      for (let index = 0; index < candidate.contourLongitude.length; index += 1) {{
        const point = project({{
          longitude: candidate.contourLongitude[index],
          latitude: candidate.contourLatitude[index],
        }});
        if (index === 0) context.moveTo(point.x, point.y);
        else context.lineTo(point.x, point.y);
      }}
      context.closePath();
      context.fill();
      context.stroke();
    }}
    context.fillStyle = stroke;
    context.beginPath();
    context.arc(center.x, center.y, 3.0, 0, Math.PI * 2);
    context.fill();
    context.strokeStyle = "#ffffff";
    context.lineWidth = 1.2;
    context.stroke();
  }}

  function drawFrame() {{
    context.strokeStyle = "#cfd8e3";
    context.lineWidth = 2;
    context.strokeRect(1, 1, canvas.width - 2, canvas.height - 2);
  }}

  function line(a, b) {{
    context.beginPath();
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.stroke();
  }}
}})();
</script>
</body>
</html>"""


def _eddy_explorer_iframe_html(document: str, height_pixels: int) -> str:
    escaped_document = html.escape(document, quote=True)
    return (
        f'<iframe srcdoc="{escaped_document}" '
        + 'style="width:100%; '
        + f'height:{height_pixels}px; border:0;" '
        + 'loading="lazy" sandbox="allow-scripts"></iframe>'
    )


def plot_multi_reference_zonal_psd_comparison(
    challenger_dataset: xarray.Dataset,
    reference_datasets: Mapping[str, xarray.Dataset],
    variables: Sequence[Variable | str] = DEFAULT_ZONAL_PSD_VARIABLES,
    first_day_index: int = 0,
    lead_day_indices: Sequence[int] = DEFAULT_ZONAL_PSD_LEAD_DAY_INDICES,
    depth_selectors: Mapping[str, int | float | None] | None = None,
    challenger_name: str = "Challenger",
):
    variable_keys = _resolved_variable_keys(variables)
    resolved_lead_day_indices = _resolved_lead_day_indices(challenger_dataset, lead_day_indices)
    if not resolved_lead_day_indices:
        raise ValueError("lead_day_indices must contain at least one available lead day index.")

    figure, axes = plt.subplots(
        1,
        len(variable_keys),
        figsize=(6.2 * len(variable_keys), 4.6),
        squeeze=False,
        constrained_layout=True,
    )
    reference_colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]
    lead_day_styles = ["-", "--", ":", "-."]

    for variable_index, variable_key in enumerate(variable_keys):
        axis = axes[0, variable_index]
        challenger_field, reference_fields = _zonal_psd_fields(
            challenger_dataset=challenger_dataset,
            reference_datasets=reference_datasets,
            variable_key=variable_key,
            first_day_index=first_day_index,
            lead_day_indices=resolved_lead_day_indices,
            depth_selector=_depth_selector_for_variable(variable_key, depth_selectors),
        )
        for lead_day_position, lead_day_index in enumerate(resolved_lead_day_indices):
            line_style = lead_day_styles[lead_day_position % len(lead_day_styles)]
            challenger_wavelength, challenger_power = _zonal_power_spectrum(
                challenger_field.isel({Dimension.LEAD_DAY_INDEX.key(): lead_day_position})
            )
            axis.plot(
                challenger_wavelength,
                challenger_power,
                color="tab:blue",
                linestyle=line_style,
                linewidth=2.0,
                label=f"{challenger_name} day {lead_day_index + 1}",
            )
            for reference_index, (_, reference_label, reference_field) in enumerate(reference_fields):
                reference_wavelength, reference_power = _zonal_power_spectrum(
                    reference_field.isel({Dimension.LEAD_DAY_INDEX.key(): lead_day_position})
                )
                axis.plot(
                    reference_wavelength,
                    reference_power,
                    color=reference_colors[reference_index % len(reference_colors)],
                    linestyle=line_style,
                    linewidth=2.0,
                    label=f"{reference_label} day {lead_day_index + 1}",
                )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.invert_xaxis()
        axis.grid(True, alpha=0.25, which="both")
        axis.set_xlabel("Zonal wavelength (km)")
        axis.set_ylabel("Power spectral density")
        axis.set_title(_variable_label_with_unit(variable_key))
        axis.legend(fontsize=8)

    figure.suptitle("Zonal power spectral density comparison", fontsize=14)
    return figure


def _resolved_variable_keys(variables: Sequence[Variable | str]) -> tuple[str, ...]:
    resolved_variable_keys = tuple(_as_variable_key(variable) for variable in variables)
    if not resolved_variable_keys:
        raise ValueError("variables must contain at least one variable.")
    return resolved_variable_keys


def _depth_selector_for_variable(
    variable_key: str,
    depth_selectors: Mapping[str, int | float | None] | None,
) -> int | float | None:
    if depth_selectors is None:
        return None
    return depth_selectors.get(variable_key)


def _depth_selectors_for_variable(
    dataset: xarray.Dataset,
    variable_key: str,
    depth_selectors: Mapping[str, int | float | None] | None,
) -> tuple[int | float | None, ...]:
    if depth_selectors is not None and variable_key in depth_selectors:
        return (_depth_selector_for_variable(variable_key, depth_selectors),)

    variable = _standard_dataset(dataset)[variable_key]
    if Dimension.DEPTH.key() not in variable.dims:
        return (None,)
    if Dimension.DEPTH.key() not in variable.coords:
        return tuple(range(min(variable.sizes[Dimension.DEPTH.key()], len(DEFAULT_SURFACE_COMPARISON_DEPTH_SELECTORS))))
    return _nearest_unique_depth_selectors(
        variable[Dimension.DEPTH.key()].values,
        DEFAULT_SURFACE_COMPARISON_DEPTH_SELECTORS,
    )


def _nearest_unique_depth_selectors(
    available_depths: numpy.ndarray,
    requested_depths: Sequence[float],
) -> tuple[float, ...]:
    available_depths = numpy.asarray(available_depths, dtype=float)
    nearest_indices = (
        int(numpy.nanargmin(numpy.abs(available_depths - requested_depth))) for requested_depth in requested_depths
    )
    unique_nearest_indices = tuple(dict.fromkeys(nearest_indices))
    return tuple(float(available_depths[index]) for index in unique_nearest_indices)


def _dataset_contains_variable(dataset: xarray.Dataset, variable_key: str) -> bool:
    return variable_key in _standard_dataset(dataset)


def _shared_multi_reference_variable_keys(
    challenger_dataset: xarray.Dataset,
    reference_datasets: Mapping[str, xarray.Dataset],
    variable_keys: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        variable_key
        for variable_key in variable_keys
        if _dataset_contains_variable(challenger_dataset, variable_key)
        and all(
            _dataset_contains_variable(reference_dataset, variable_key)
            for reference_dataset in reference_datasets.values()
        )
    )


def _surface_comparison_multi_reference_variable_payloads(
    challenger_dataset: xarray.Dataset,
    reference_datasets: Mapping[str, xarray.Dataset],
    challenger_name: str,
    variable_keys: Sequence[str],
    first_day_index: int,
    lead_day_indices: Sequence[int],
    depth_selectors: Mapping[str, int | float | None] | None,
    maximum_map_cells: int,
) -> list[dict[str, object]]:
    variable_payloads = []
    for variable_key in _shared_multi_reference_variable_keys(challenger_dataset, reference_datasets, variable_keys):
        depth_payloads = []
        for depth_index, depth_selector in enumerate(
            _depth_selectors_for_variable(challenger_dataset, variable_key, depth_selectors)
        ):
            fields = _multi_reference_comparison_fields(
                challenger_dataset=challenger_dataset,
                reference_datasets=reference_datasets,
                variable_key=variable_key,
                first_day_index=first_day_index,
                lead_day_indices=lead_day_indices,
                depth_selector=depth_selector,
                maximum_map_cells=maximum_map_cells,
            )
            depth_payloads.append(
                _surface_comparison_multi_reference_depth_payload(
                    fields,
                    variable_key=variable_key,
                    challenger_name=challenger_name,
                    depth_key=f"depth_{depth_index}",
                )
            )
        variable_payloads.append(
            _surface_comparison_variable_payload(
                variable_key=variable_key,
                depth_payloads=depth_payloads,
            )
        )
    if not variable_payloads:
        raise ValueError("None of the requested variables are available in the challenger and reference datasets.")
    return variable_payloads


def plot_surface_comparison_explorer(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    reference_name: str,
    variables: Sequence[Variable | str] = DEFAULT_SURFACE_COMPARISON_VARIABLES,
    first_day_index: int = 0,
    lead_day_indices: Sequence[int] | None = None,
    depth_selectors: Mapping[str, int | float | None] | None = None,
    challenger_name: str = "Challenger",
    maximum_map_cells: int = DEFAULT_EXPLORER_MAXIMUM_MAP_CELLS,
    height_pixels: int = DEFAULT_EXPLORER_HEIGHT_PIXELS,
    title: str = "Forecast comparison maps",
) -> HTML:
    return plot_multi_reference_surface_comparison_explorer(
        challenger_dataset=challenger_dataset,
        reference_datasets={reference_name: reference_dataset},
        variables=variables,
        first_day_index=first_day_index,
        lead_day_indices=lead_day_indices,
        depth_selectors=depth_selectors,
        challenger_name=challenger_name,
        maximum_map_cells=maximum_map_cells,
        height_pixels=height_pixels,
        title=title,
    )


def plot_multi_reference_surface_comparison_explorer(
    challenger_dataset: xarray.Dataset,
    reference_datasets: Mapping[str, xarray.Dataset],
    variables: Sequence[Variable | str] = DEFAULT_SURFACE_COMPARISON_VARIABLES,
    first_day_index: int = 0,
    lead_day_indices: Sequence[int] | None = None,
    depth_selectors: Mapping[str, int | float | None] | None = None,
    challenger_name: str = "Challenger",
    maximum_map_cells: int = DEFAULT_EXPLORER_MAXIMUM_MAP_CELLS,
    height_pixels: int = DEFAULT_EXPLORER_HEIGHT_PIXELS,
    title: str = "Forecast comparison maps",
) -> HTML:
    variable_keys = _resolved_variable_keys(variables)
    if lead_day_indices is None:
        resolved_lead_day_indices = _all_lead_day_indices(challenger_dataset)
    else:
        resolved_lead_day_indices = tuple(lead_day_indices)
    if not resolved_lead_day_indices:
        raise ValueError("lead_day_indices must contain at least one lead day index.")

    variable_payloads = _surface_comparison_multi_reference_variable_payloads(
        challenger_dataset=challenger_dataset,
        reference_datasets=reference_datasets,
        challenger_name=challenger_name,
        variable_keys=variable_keys,
        first_day_index=first_day_index,
        lead_day_indices=resolved_lead_day_indices,
        depth_selectors=depth_selectors,
        maximum_map_cells=maximum_map_cells,
    )
    element_id = f"ob-surface-comparison-{uuid4().hex}"
    payload = _surface_comparison_payload(
        reference_name=None,
        variable_payloads=variable_payloads,
        title=title,
    )
    document = _surface_comparison_explorer_document(element_id, payload)
    return _html_without_iframe_warning(_surface_comparison_iframe_html(document, height_pixels=height_pixels))


def plot_multi_reference_lagrangian_trajectory_explorer(
    challenger_dataset: xarray.Dataset,
    reference_datasets: Mapping[str, xarray.Dataset],
    first_day_index: int = 0,
    particle_count: int = DEFAULT_LAGRANGIAN_PARTICLE_COUNT,
    seed: int = 123,
    challenger_name: str = "Challenger",
    height_pixels: int = DEFAULT_EXPLORER_HEIGHT_PIXELS,
    title: str = "Lagrangian trajectory divergence",
) -> HTML:
    payload = _lagrangian_payload(
        challenger_dataset=challenger_dataset,
        reference_datasets=reference_datasets,
        first_day_index=first_day_index,
        particle_count=particle_count,
        seed=seed,
        challenger_name=challenger_name,
        title=title,
    )
    element_id = f"ob-lagrangian-{uuid4().hex}"
    document = _lagrangian_explorer_document(element_id, payload)
    return _html_without_iframe_warning(_lagrangian_explorer_iframe_html(document, height_pixels=height_pixels))


def plot_multi_reference_eddy_matching_explorer(
    challenger_dataset: xarray.Dataset,
    reference_datasets: Mapping[str, xarray.Dataset],
    first_day_index: int = 0,
    lead_day_indices: Sequence[int] | None = None,
    maximum_contour_points: int = DEFAULT_EDDY_MAXIMUM_CONTOUR_POINTS,
    height_pixels: int = DEFAULT_EXPLORER_HEIGHT_PIXELS,
    title: str = "Mesoscale eddy matching",
) -> HTML:
    payload = _eddy_payload(
        challenger_dataset=challenger_dataset,
        reference_datasets=reference_datasets,
        first_day_index=first_day_index,
        lead_day_indices=lead_day_indices,
        maximum_contour_points=maximum_contour_points,
        title=title,
        detection_parameters=eddies.default_eddy_detection_parameters(),
    )
    element_id = f"ob-eddy-{uuid4().hex}"
    document = _eddy_explorer_document(element_id, payload)
    return _html_without_iframe_warning(_eddy_explorer_iframe_html(document, height_pixels=height_pixels))
