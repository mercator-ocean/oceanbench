// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

const DEFAULT_REMOTE_DATA_BASE_URL =
  "https://minio.dive.edito.eu/project-oceanbench/dev/benchmark/rebuild-preview/viewer/data/";

function normalizeBaseUrl(url) {
  return url.endsWith("/") ? url : `${url}/`;
}

function configuredDataBaseUrl() {
  const parameters = new URLSearchParams(window.location.search);
  const queryValue = parameters.get("data_base") || parameters.get("dataBaseUrl");
  if (queryValue === "local") return "./data/";
  return window.OCEANBENCH_VIEWER_CONFIG?.dataBaseUrl || queryValue || DEFAULT_REMOTE_DATA_BASE_URL;
}

export const DATA_BASE_URL = new URL(
  normalizeBaseUrl(configuredDataBaseUrl()),
  window.location.href,
).href;

window.__oceanbenchViewerDataBaseUrl = DATA_BASE_URL;

export function resolveViewerDataUrl(url) {
  if (!url || /^(?:[a-z]+:)?\/\//i.test(url) || url.startsWith("data:")) return url;
  const withoutDataPrefix = url.replace(/^\.?\/*data\//, "");
  return new URL(withoutDataPrefix, DATA_BASE_URL).href;
}

// Water-column stores (<slug>.columns.zarr) may live beside the pyramids or at a
// separate location (they are large and published independently). They default to the
// same base as the pyramids; a `columns_base` query param (or window config) overrides
// just the column store, so a developer can point the pyramids at the live bucket while
// serving a locally-built column store from a local server.
function configuredColumnsBaseUrl() {
  const parameters = new URLSearchParams(window.location.search);
  const queryValue = parameters.get("columns_base");
  if (queryValue === "local") return "./data/";
  return window.OCEANBENCH_VIEWER_CONFIG?.columnsBaseUrl || queryValue || DATA_BASE_URL;
}

export const COLUMNS_BASE_URL = new URL(
  normalizeBaseUrl(configuredColumnsBaseUrl()),
  window.location.href,
).href;

// The column store conventionally sits beside the pyramid as `<slug>.columns.zarr`.
export function resolveColumnStoreUrl(slug) {
  return new URL(`${slug}.columns.zarr`, COLUMNS_BASE_URL).href;
}
