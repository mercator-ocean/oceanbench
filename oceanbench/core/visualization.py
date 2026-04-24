# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Mapping, Sequence
from base64 import b64encode
import html
import json
from uuid import uuid4

from IPython.display import HTML
from matplotlib import colormaps
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.colors import to_hex
import numpy
import xarray

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable, VARIABLE_LABELS, VARIABLE_METADATA

DEFAULT_SURFACE_COMPARISON_VARIABLES: tuple[Variable, ...] = (
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
    Variable.SEA_WATER_SALINITY,
    Variable.EASTWARD_SEA_WATER_VELOCITY,
    Variable.NORTHWARD_SEA_WATER_VELOCITY,
)
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
QUANTIZED_MISSING_VALUE = -32768
QUANTIZED_MINIMUM_VALUE = -32767
QUANTIZED_MAXIMUM_VALUE = 32767
DEFAULT_EXPLORER_MAXIMUM_MAP_CELLS = 160_000
DEFAULT_EXPLORER_HEIGHT_PIXELS = 760


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


def _select_surface_field(
    dataset: xarray.Dataset,
    variable_key: str,
    first_day_index: int,
    lead_day_index: int,
    depth_selector: int | float | None,
) -> xarray.DataArray:
    standard_dataset = _standard_dataset(dataset)
    if variable_key not in standard_dataset:
        raise ValueError(f"Dataset does not contain variable {variable_key!r}.")

    field = standard_dataset[variable_key]
    field = _select_dimension_index(field, Dimension.FIRST_DAY_DATETIME.key(), first_day_index)
    field = _select_dimension_index(field, Dimension.LEAD_DAY_INDEX.key(), lead_day_index)
    field = _select_depth(field, depth_selector)
    return field


def _select_surface_lead_field(
    dataset: xarray.Dataset,
    variable_key: str,
    lead_day_index: int,
    depth_selector: int | float | None,
) -> xarray.DataArray:
    standard_dataset = _standard_dataset(dataset)
    if variable_key not in standard_dataset:
        raise ValueError(f"Dataset does not contain variable {variable_key!r}.")

    field = standard_dataset[variable_key]
    field = _select_dimension_index(field, Dimension.LEAD_DAY_INDEX.key(), lead_day_index)
    field = _select_depth(field, depth_selector)
    return field


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


def _comparison_fields(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variable_key: str,
    first_day_index: int,
    lead_day_indices: Sequence[int],
    depth_selector: int | float | None,
    maximum_map_cells: int,
) -> tuple[
    list[tuple[int, xarray.DataArray, xarray.DataArray, xarray.DataArray, xarray.DataArray]],
    int,
]:
    fields = []
    spatial_stride = 1
    for lead_day_index in lead_day_indices:
        challenger_field = _select_surface_field(
            challenger_dataset,
            variable_key,
            first_day_index,
            lead_day_index,
            depth_selector,
        )
        reference_field = _select_surface_field(
            reference_dataset,
            variable_key,
            first_day_index,
            lead_day_index,
            depth_selector,
        )
        challenger_field, reference_field = xarray.align(challenger_field, reference_field, join="inner")
        if not fields:
            spatial_stride = _spatial_stride(challenger_field, maximum_map_cells)
        error_field = challenger_field - reference_field
        rmse_field = _rmse_over_forecast_dates_field(
            challenger_dataset,
            reference_dataset,
            variable_key,
            lead_day_index,
            depth_selector,
        )
        fields.append(
            (
                lead_day_index,
                _thin_field(challenger_field, spatial_stride).compute(),
                _thin_field(reference_field, spatial_stride).compute(),
                _thin_field(error_field, spatial_stride).compute(),
                _thin_field(rmse_field, spatial_stride).compute(),
            )
        )
    return fields, spatial_stride


def _rmse_over_forecast_dates_field(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variable_key: str,
    lead_day_index: int,
    depth_selector: int | float | None,
) -> xarray.DataArray:
    challenger_field = _select_surface_lead_field(
        challenger_dataset,
        variable_key,
        lead_day_index,
        depth_selector,
    )
    reference_field = _select_surface_lead_field(
        reference_dataset,
        variable_key,
        lead_day_index,
        depth_selector,
    )
    challenger_field, reference_field = xarray.align(challenger_field, reference_field, join="inner")
    error_field = challenger_field - reference_field
    reduction_dimensions = [
        Dimension.FIRST_DAY_DATETIME.key(),
    ]
    reduction_dimensions = [dimension for dimension in reduction_dimensions if dimension in error_field.dims]
    if not reduction_dimensions:
        return abs(error_field)
    return numpy.sqrt((error_field**2).mean(dim=reduction_dimensions))


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


def _sample_colormap(colormap_name: str, color_count: int = 256) -> list[str]:
    colormap = colormaps[colormap_name]
    return [to_hex(colormap(index / (color_count - 1)), keep_alpha=False) for index in range(color_count)]


def _norm_limits(norm: Normalize) -> tuple[float, float]:
    return float(norm.vmin), float(norm.vmax)


def _quantize_array(array: numpy.ndarray, minimum: float, maximum: float) -> tuple[numpy.ndarray, float, float]:
    if maximum <= minimum:
        maximum = minimum + 1.0
    scale = (maximum - minimum) / (QUANTIZED_MAXIMUM_VALUE - QUANTIZED_MINIMUM_VALUE)
    offset = minimum - QUANTIZED_MINIMUM_VALUE * scale
    values = numpy.asarray(array, dtype=float)
    finite_mask = numpy.isfinite(values)
    quantized = numpy.full(values.shape, QUANTIZED_MISSING_VALUE, dtype="<i2")
    clipped_values = numpy.clip(values[finite_mask], minimum, maximum)
    quantized[finite_mask] = numpy.rint((clipped_values - offset) / scale).astype("<i2")
    return quantized, float(scale), float(offset)


def _encoded_layer(
    key: str,
    label: str,
    arrays: Sequence[xarray.DataArray],
    norm: Normalize,
    colormap_name: str,
    value_label: str,
) -> dict[str, object]:
    stacked_values = numpy.stack([numpy.asarray(array.values, dtype=float) for array in arrays])
    minimum, maximum = _norm_limits(norm)
    quantized_values, scale, offset = _quantize_array(stacked_values, minimum, maximum)
    return {
        "key": key,
        "label": label,
        "data": b64encode(quantized_values.tobytes()).decode("ascii"),
        "scale": scale,
        "offset": offset,
        "missing": QUANTIZED_MISSING_VALUE,
        "minimum": minimum,
        "maximum": maximum,
        "colormap": _sample_colormap(colormap_name),
        "valueLabel": value_label,
    }


def _rounded_coordinate_values(values: numpy.ndarray) -> list[float]:
    return [round(float(value), 6) for value in values]


def _surface_comparison_depth_payload(
    fields: Sequence[tuple[int, xarray.DataArray, xarray.DataArray, xarray.DataArray, xarray.DataArray]],
    variable_key: str,
    reference_name: str,
    challenger_name: str,
    depth_key: str,
    spatial_stride: int,
) -> dict[str, object]:
    challenger_fields = [challenger_field for _, challenger_field, _, _, _ in fields]
    reference_fields = [reference_field for _, _, reference_field, _, _ in fields]
    error_fields = [error_field for _, _, _, error_field, _ in fields]
    absolute_error_fields = [abs(error_field) for error_field in error_fields]
    rmse_fields = [rmse_field for _, _, _, _, rmse_field in fields]
    field_norm = _field_norm(variable_key, [*challenger_fields, *reference_fields])
    error_norm = _symmetric_norm(error_fields)
    absolute_error_norm = _positive_norm(absolute_error_fields)
    rmse_norm = _positive_norm(rmse_fields)
    latitude_values = challenger_fields[0][Dimension.LATITUDE.key()].values
    longitude_values = challenger_fields[0][Dimension.LONGITUDE.key()].values

    return {
        "key": depth_key,
        "label": _depth_label(challenger_fields[0]),
        "leadDays": [
            _lead_day_label(challenger_field, lead_day_index) for lead_day_index, challenger_field, _, _, _ in fields
        ],
        "latitude": _rounded_coordinate_values(latitude_values),
        "longitude": _rounded_coordinate_values(longitude_values),
        "shape": [len(fields), len(latitude_values), len(longitude_values)],
        "spatialStride": spatial_stride,
        "layers": [
            _encoded_layer(
                "challenger",
                challenger_name,
                challenger_fields,
                field_norm,
                _field_colormap(variable_key),
                _variable_label_with_unit(variable_key),
            ),
            _encoded_layer(
                "reference",
                reference_name,
                reference_fields,
                field_norm,
                _field_colormap(variable_key),
                _variable_label_with_unit(variable_key),
            ),
            _encoded_layer(
                "error",
                "Signed error",
                error_fields,
                error_norm,
                ERROR_COLORMAP,
                f"{_variable_label_with_unit(variable_key)} error",
            ),
            _encoded_layer(
                "absolute_error",
                "Absolute error",
                absolute_error_fields,
                absolute_error_norm,
                ABSOLUTE_ERROR_COLORMAP,
                f"{_variable_label_with_unit(variable_key)} absolute error",
            ),
            _encoded_layer(
                "rmse_over_dates",
                "RMSE over dates",
                rmse_fields,
                rmse_norm,
                RMSE_COLORMAP,
                f"{_variable_label_with_unit(variable_key)} RMSE",
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
    reference_name: str,
) -> dict[str, object]:
    return {
        "title": f"Surface comparison against {reference_name}",
        "landColor": LAND_BACKGROUND_COLOR,
        "axisColor": "#334155",
        "gridColor": "rgba(71, 85, 105, 0.22)",
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
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  justify-content: flex-end;
  gap: 10px 14px;
}}
.ob-map-variable-buttons,
.ob-map-depth-buttons,
.ob-map-layer-buttons {{
  display: inline-flex;
  max-width: 100%;
  overflow-x: auto;
  border: 1px solid #cbd5e1;
  border-radius: 7px;
  background: #ffffff;
}}
.ob-map-variable-button,
.ob-map-depth-button,
.ob-map-layer-button {{
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
.ob-map-variable-button:last-child,
.ob-map-depth-button:last-child,
.ob-map-layer-button:last-child {{
  border-right: 0;
}}
.ob-map-variable-button.active,
.ob-map-depth-button.active,
.ob-map-layer-button.active {{
  background: #0f5f8f;
  color: #ffffff;
}}
.ob-map-depth-buttons.hidden {{
  display: none;
}}
.ob-map-lead-control {{
  display: grid;
  grid-template-columns: 12ch minmax(150px, 240px);
  align-items: center;
  gap: 8px;
  color: #334155;
  font-size: 13px;
}}
.ob-map-lead-label {{
  display: block;
  min-width: 12ch;
  text-align: right;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}}
.ob-map-lead-control input {{
  accent-color: #0f5f8f;
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
.ob-map-canvas,
.ob-map-colorbar {{
  display: block;
  width: 100%;
}}
.ob-map-canvas {{
  height: 100%;
}}
.ob-map-colorbar {{
  height: 58px;
  margin-top: 8px;
}}
.ob-map-status {{
  min-height: 18px;
  margin-top: 2px;
  color: #475569;
  font-size: 12px;
}}
@media (max-width: 760px) {{
  .ob-map-header {{
    display: block;
  }}
  .ob-map-controls {{
    justify-content: flex-start;
    margin-top: 10px;
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
      <div class="ob-map-variable-buttons"></div>
      <div class="ob-map-depth-buttons"></div>
      <div class="ob-map-layer-buttons"></div>
      <label class="ob-map-lead-control">
        <span class="ob-map-lead-label"></span>
        <input class="ob-map-lead-input" type="range">
      </label>
    </div>
  </div>
  <div class="ob-map-canvas-wrap">
    <canvas class="ob-map-canvas"></canvas>
  </div>
  <canvas class="ob-map-colorbar"></canvas>
  <div class="ob-map-status"></div>
</div>
<script>
(() => {{
  const payload = {payload_json};
  const root = document.getElementById("{element_id}");
  const title = root.querySelector(".ob-map-title");
  const variableButtons = root.querySelector(".ob-map-variable-buttons");
  const depthButtons = root.querySelector(".ob-map-depth-buttons");
  const layerButtons = root.querySelector(".ob-map-layer-buttons");
  const leadLabel = root.querySelector(".ob-map-lead-label");
  const leadInput = root.querySelector(".ob-map-lead-input");
  const canvas = root.querySelector(".ob-map-canvas");
  const colorbar = root.querySelector(".ob-map-colorbar");
  const status = root.querySelector(".ob-map-status");
  const context = canvas.getContext("2d");
  const colorbarContext = colorbar.getContext("2d");
  const variables = Object.fromEntries(payload.variables.map((variable) => [variable.key, variable]));
  const variableData = Object.fromEntries(
    payload.variables.map((variable) => [
      variable.key,
      Object.fromEntries(
        variable.depths.map((depth) => [
          depth.key,
          Object.fromEntries(depth.layers.map((layer) => [layer.key, decodeInt16(layer.data)])),
        ]),
      ),
    ]),
  );
  let activeVariableKey = payload.variables[0].key;
  let activeDepthKey = payload.variables[0].depths[0].key;
  let activeLayerKey = "error";
  let activeLeadIndex = 0;
  let plotArea = {{ left: 0, top: 0, width: 0, height: 0 }};

  title.textContent = payload.title;

  for (const variable of payload.variables) {{
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ob-map-variable-button";
    button.textContent = variable.label;
    button.dataset.variableKey = variable.key;
    button.addEventListener("click", () => {{
      activeVariableKey = variable.key;
      activeDepthKey = currentVariable().depths[0].key;
      activeLeadIndex = Math.min(activeLeadIndex, currentDepth().shape[0] - 1);
      buildDepthButtons();
      updateControls();
      render();
    }});
    variableButtons.appendChild(button);
  }}

  buildDepthButtons();

  for (const layer of payload.variables[0].depths[0].layers) {{
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ob-map-layer-button";
    button.textContent = layer.label;
    button.dataset.layerKey = layer.key;
    button.addEventListener("click", () => {{
      activeLayerKey = layer.key;
      updateControls();
      render();
    }});
    layerButtons.appendChild(button);
  }}

  leadInput.addEventListener("input", () => {{
    activeLeadIndex = Number(leadInput.value);
    render();
  }});
  canvas.addEventListener("mousemove", updateStatus);
  canvas.addEventListener("mouseleave", () => {{
    status.textContent = "";
  }});
  window.addEventListener("resize", render);

  updateControls();
  render();

  function decodeInt16(base64) {{
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {{
      bytes[index] = binary.charCodeAt(index);
    }}
    return new Int16Array(bytes.buffer);
  }}

  function layerValue(layer, storedValue) {{
    if (storedValue === layer.missing) {{
      return Number.NaN;
    }}
    return storedValue * layer.scale + layer.offset;
  }}

  function dataIndex(variable, leadIndex, latitudeIndex, longitudeIndex) {{
    const [, latitudeCount, longitudeCount] = variable.shape;
    return leadIndex * latitudeCount * longitudeCount + latitudeIndex * longitudeCount + longitudeIndex;
  }}

  function currentVariable() {{
    return variables[activeVariableKey];
  }}

  function currentDepth() {{
    return Object.fromEntries(currentVariable().depths.map((depth) => [depth.key, depth]))[activeDepthKey];
  }}

  function currentLayer() {{
    return Object.fromEntries(currentDepth().layers.map((layer) => [layer.key, layer]))[activeLayerKey];
  }}

  function currentLayerData() {{
    return variableData[activeVariableKey][activeDepthKey][activeLayerKey];
  }}

  function buildDepthButtons() {{
    depthButtons.replaceChildren();
    const variable = currentVariable();
    depthButtons.classList.toggle("hidden", variable.depths.length <= 1);
    for (const depth of variable.depths) {{
      const button = document.createElement("button");
      button.type = "button";
      button.className = "ob-map-depth-button";
      button.textContent = depth.label;
      button.dataset.depthKey = depth.key;
      button.addEventListener("click", () => {{
        activeDepthKey = depth.key;
        activeLeadIndex = Math.min(activeLeadIndex, currentDepth().shape[0] - 1);
        updateControls();
        render();
      }});
      depthButtons.appendChild(button);
    }}
  }}

  function updateControls() {{
    const variable = currentVariable();
    if (!variable.depths.some((depth) => depth.key === activeDepthKey)) {{
      activeDepthKey = variable.depths[0].key;
    }}
    const depth = currentDepth();
    leadInput.min = 0;
    leadInput.max = depth.shape[0] - 1;
    leadInput.step = 1;
    leadInput.value = activeLeadIndex;
    for (const button of variableButtons.querySelectorAll("button")) {{
      button.classList.toggle("active", button.dataset.variableKey === activeVariableKey);
    }}
    for (const button of depthButtons.querySelectorAll("button")) {{
      button.classList.toggle("active", button.dataset.depthKey === activeDepthKey);
    }}
    for (const button of layerButtons.querySelectorAll("button")) {{
      button.classList.toggle("active", button.dataset.layerKey === activeLayerKey);
    }}
  }}

  function setupCanvas(targetCanvas, targetContext, cssHeight) {{
    const deviceRatio = window.devicePixelRatio || 1;
    const width = Math.max(320, Math.floor(targetCanvas.clientWidth));
    const height = Math.max(cssHeight, Math.floor(targetCanvas.clientHeight));
    targetCanvas.width = Math.floor(width * deviceRatio);
    targetCanvas.height = Math.floor(height * deviceRatio);
    targetContext.setTransform(deviceRatio, 0, 0, deviceRatio, 0, 0);
    return {{ width, height }};
  }}

  function render() {{
    const size = setupCanvas(canvas, context, 420);
    const depth = currentDepth();
    const layer = currentLayer();
    const values = currentLayerData();
    leadLabel.textContent = `Lead day ${{depth.leadDays[activeLeadIndex]}}`;
    context.clearRect(0, 0, size.width, size.height);
    drawMap(context, size, depth, layer, values);
    drawColorbar(layer);
  }}

  function drawMap(targetContext, size, depth, layer, values) {{
    const margin = {{ left: 56, top: 18, right: 16, bottom: 40 }};
    plotArea = {{
      left: margin.left,
      top: margin.top,
      width: size.width - margin.left - margin.right,
      height: size.height - margin.top - margin.bottom,
    }};
    const [, latitudeCount, longitudeCount] = depth.shape;
    const latitude = depth.latitude;
    const longitude = depth.longitude;
    const cellWidth = plotArea.width / longitudeCount;
    const cellHeight = plotArea.height / latitudeCount;
    targetContext.fillStyle = "#ffffff";
    targetContext.fillRect(0, 0, size.width, size.height);
    targetContext.fillStyle = payload.landColor;
    targetContext.fillRect(plotArea.left, plotArea.top, plotArea.width, plotArea.height);

    const latitudeAscending = latitude[0] < latitude[latitude.length - 1];
    for (let latitudeIndex = 0; latitudeIndex < latitudeCount; latitudeIndex += 1) {{
      const yIndex = latitudeAscending ? latitudeCount - 1 - latitudeIndex : latitudeIndex;
      const y = plotArea.top + yIndex * cellHeight;
      for (let longitudeIndex = 0; longitudeIndex < longitudeCount; longitudeIndex += 1) {{
        const value = layerValue(
          layer,
          values[dataIndex(depth, activeLeadIndex, latitudeIndex, longitudeIndex)],
        );
        targetContext.fillStyle = Number.isFinite(value) ? colorForValue(layer, value) : payload.landColor;
        targetContext.fillRect(
          plotArea.left + longitudeIndex * cellWidth,
          y,
          Math.ceil(cellWidth) + 0.5,
          Math.ceil(cellHeight) + 0.5,
        );
      }}
    }}
    drawAxes(targetContext, depth);
  }}

  function colorForValue(layer, value) {{
    const fraction = Math.max(0, Math.min(1, (value - layer.minimum) / (layer.maximum - layer.minimum)));
    const colorIndex = Math.round(fraction * (layer.colormap.length - 1));
    return layer.colormap[colorIndex];
  }}

  function niceStep(span, targetCount) {{
    const roughStep = span / targetCount;
    const power = Math.pow(10, Math.floor(Math.log10(roughStep)));
    const normalized = roughStep / power;
    if (normalized <= 1) return power;
    if (normalized <= 2) return 2 * power;
    if (normalized <= 5) return 5 * power;
    return 10 * power;
  }}

  function ticks(minimum, maximum, targetCount) {{
    const step = niceStep(maximum - minimum, targetCount);
    const first = Math.ceil(minimum / step) * step;
    const values = [];
    for (let value = first; value <= maximum + step * 0.5; value += step) {{
      values.push(Math.abs(value) < 1e-9 ? 0 : value);
    }}
    return values;
  }}

  function drawAxes(targetContext, depth) {{
    const latitude = depth.latitude;
    const longitude = depth.longitude;
    const longitudeMinimum = Math.min(...longitude);
    const longitudeMaximum = Math.max(...longitude);
    const latitudeMinimum = Math.min(...latitude);
    const latitudeMaximum = Math.max(...latitude);
    const longitudeTicks = ticks(longitudeMinimum, longitudeMaximum, 7);
    const latitudeTicks = ticks(latitudeMinimum, latitudeMaximum, 5);
    const latitudeAscending = latitude[0] < latitude[latitude.length - 1];

    targetContext.save();
    targetContext.strokeStyle = payload.gridColor;
    targetContext.fillStyle = payload.axisColor;
    targetContext.lineWidth = 1;
    targetContext.font = "12px ui-sans-serif, system-ui, sans-serif";
    targetContext.textAlign = "center";
    targetContext.textBaseline = "top";
    for (const tick of longitudeTicks) {{
      const x = plotArea.left + ((tick - longitudeMinimum) / (longitudeMaximum - longitudeMinimum)) * plotArea.width;
      targetContext.beginPath();
      targetContext.moveTo(x, plotArea.top);
      targetContext.lineTo(x, plotArea.top + plotArea.height);
      targetContext.stroke();
      targetContext.fillText(formatTick(tick), x, plotArea.top + plotArea.height + 8);
    }}

    targetContext.textAlign = "right";
    targetContext.textBaseline = "middle";
    for (const tick of latitudeTicks) {{
      const fraction = (tick - latitudeMinimum) / (latitudeMaximum - latitudeMinimum);
      const y = latitudeAscending
        ? plotArea.top + (1 - fraction) * plotArea.height
        : plotArea.top + fraction * plotArea.height;
      targetContext.beginPath();
      targetContext.moveTo(plotArea.left, y);
      targetContext.lineTo(plotArea.left + plotArea.width, y);
      targetContext.stroke();
      targetContext.fillText(formatTick(tick), plotArea.left - 8, y);
    }}

    targetContext.strokeStyle = "#1f2937";
    targetContext.strokeRect(plotArea.left, plotArea.top, plotArea.width, plotArea.height);
    targetContext.restore();
  }}

  function formatTick(value) {{
    return Number.isInteger(value) ? String(value) : value.toFixed(1);
  }}

  function drawColorbar(layer) {{
    const size = setupCanvas(colorbar, colorbarContext, 58);
    const margin = {{ left: 56, top: 8, right: 16, bottom: 28 }};
    const barWidth = size.width - margin.left - margin.right;
    const barHeight = 14;
    colorbarContext.clearRect(0, 0, size.width, size.height);
    for (let index = 0; index < barWidth; index += 1) {{
      const fraction = index / Math.max(1, barWidth - 1);
      const colorIndex = Math.round(fraction * (layer.colormap.length - 1));
      colorbarContext.fillStyle = layer.colormap[colorIndex];
      colorbarContext.fillRect(margin.left + index, margin.top, 1, barHeight);
    }}
    colorbarContext.strokeStyle = "#1f2937";
    colorbarContext.strokeRect(margin.left, margin.top, barWidth, barHeight);
    colorbarContext.fillStyle = payload.axisColor;
    colorbarContext.font = "12px ui-sans-serif, system-ui, sans-serif";
    colorbarContext.textAlign = "left";
    colorbarContext.textBaseline = "top";
    colorbarContext.fillText(layer.minimum.toPrecision(3), margin.left, margin.top + barHeight + 6);
    colorbarContext.textAlign = "center";
    colorbarContext.fillText(layer.valueLabel, margin.left + barWidth / 2, margin.top + barHeight + 6);
    colorbarContext.textAlign = "right";
    colorbarContext.fillText(layer.maximum.toPrecision(3), margin.left + barWidth, margin.top + barHeight + 6);
  }}

  function updateStatus(event) {{
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (
      x < plotArea.left ||
      x > plotArea.left + plotArea.width ||
      y < plotArea.top ||
      y > plotArea.top + plotArea.height
    ) {{
      status.textContent = "";
      return;
    }}
    const variable = currentVariable();
    const depth = currentDepth();
    const layer = currentLayer();
    const values = currentLayerData();
    const [, latitudeCount, longitudeCount] = depth.shape;
    const latitude = depth.latitude;
    const longitude = depth.longitude;
    const longitudeIndex = Math.max(
      0,
      Math.min(longitudeCount - 1, Math.floor(((x - plotArea.left) / plotArea.width) * longitudeCount)),
    );
    const latitudeAscending = latitude[0] < latitude[latitude.length - 1];
    const yIndex = Math.max(
      0,
      Math.min(latitudeCount - 1, Math.floor(((y - plotArea.top) / plotArea.height) * latitudeCount)),
    );
    const latitudeIndex = latitudeAscending ? latitudeCount - 1 - yIndex : yIndex;
    const value = layerValue(
      layer,
      values[dataIndex(depth, activeLeadIndex, latitudeIndex, longitudeIndex)],
    );
    const valueText = Number.isFinite(value) ? value.toPrecision(4) : "land / no data";
    const depthText = currentVariable().depths.length > 1 ? ` · ${{depth.label}}` : "";
    status.textContent = [
      `lon ${{longitude[longitudeIndex].toFixed(2)}}`,
      `lat ${{latitude[latitudeIndex].toFixed(2)}}`,
      `${{variable.label}}${{depthText}}`,
      `${{layer.label}}: ${{valueText}}`,
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
        return tuple(range(variable.sizes[Dimension.DEPTH.key()]))
    return tuple(float(depth) for depth in variable[Dimension.DEPTH.key()].values)


def _dataset_contains_variable(dataset: xarray.Dataset, variable_key: str) -> bool:
    return variable_key in _standard_dataset(dataset)


def _shared_variable_keys(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variable_keys: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        variable_key
        for variable_key in variable_keys
        if _dataset_contains_variable(challenger_dataset, variable_key)
        and _dataset_contains_variable(reference_dataset, variable_key)
    )


def _surface_comparison_variable_payloads(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    reference_name: str,
    challenger_name: str,
    variable_keys: Sequence[str],
    first_day_index: int,
    lead_day_indices: Sequence[int],
    depth_selectors: Mapping[str, int | float | None] | None,
    maximum_map_cells: int,
) -> list[dict[str, object]]:
    variable_payloads = []
    for variable_key in _shared_variable_keys(challenger_dataset, reference_dataset, variable_keys):
        depth_payloads = []
        for depth_index, depth_selector in enumerate(
            _depth_selectors_for_variable(challenger_dataset, variable_key, depth_selectors)
        ):
            fields, spatial_stride = _comparison_fields(
                challenger_dataset=challenger_dataset,
                reference_dataset=reference_dataset,
                variable_key=variable_key,
                first_day_index=first_day_index,
                lead_day_indices=lead_day_indices,
                depth_selector=depth_selector,
                maximum_map_cells=maximum_map_cells,
            )
            depth_payloads.append(
                _surface_comparison_depth_payload(
                    fields,
                    variable_key=variable_key,
                    reference_name=reference_name,
                    challenger_name=challenger_name,
                    depth_key=f"depth_{depth_index}",
                    spatial_stride=spatial_stride,
                )
            )
        variable_payloads.append(
            _surface_comparison_variable_payload(
                variable_key=variable_key,
                depth_payloads=depth_payloads,
            )
        )
    if not variable_payloads:
        raise ValueError("None of the requested variables are available in both datasets.")
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
) -> HTML:
    variable_keys = _resolved_variable_keys(variables)
    if lead_day_indices is None:
        resolved_lead_day_indices = _all_lead_day_indices(challenger_dataset)
    else:
        resolved_lead_day_indices = tuple(lead_day_indices)
    if not resolved_lead_day_indices:
        raise ValueError("lead_day_indices must contain at least one lead day index.")

    variable_payloads = _surface_comparison_variable_payloads(
        challenger_dataset=challenger_dataset,
        reference_dataset=reference_dataset,
        reference_name=reference_name,
        challenger_name=challenger_name,
        variable_keys=variable_keys,
        first_day_index=first_day_index,
        lead_day_indices=resolved_lead_day_indices,
        depth_selectors=depth_selectors,
        maximum_map_cells=maximum_map_cells,
    )
    element_id = f"ob-surface-comparison-{uuid4().hex}"
    payload = _surface_comparison_payload(
        reference_name=reference_name,
        variable_payloads=variable_payloads,
    )
    document = _surface_comparison_explorer_document(element_id, payload)
    return HTML(_surface_comparison_iframe_html(document, height_pixels=height_pixels))
