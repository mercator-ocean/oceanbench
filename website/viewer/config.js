// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// CROSS-PAGE CONTRACT: website/scores-summary.js imports initializeViewerConfig and
// viewerDataBaseUrl from here, so the data-root precedence below is the scores page's as
// much as the viewer's. Keep the two in one place; do not fork it. The file stays under
// website/viewer/ because that directory is also published on its own.

const DEFAULT_REMOTE_DATA_BASE_URL =
  "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/dev/benchmark/rebuild-preview/viewer/data/";

// Optional side-car file, fetched once at startup and 404-tolerated, so a static deployment
// can pin a data root without editing this file: drop it beside index.html. `oceanbench view`
// writes no such file, it mounts the artifacts directory and points the viewer at it with
// `?data=/data/`. Query parameters still win over the side-car, and with neither present the
// default stays the published bucket prefix above. The published CloudFerro copy ships such a
// side-car beside its index.html, which is why that deployment reads data from its own host
// even when this built-in default changes.
const VIEWER_CONFIG_FILE = "./viewer-config.json";

let fileConfig = null;

function normalizeBaseUrl(url) {
  return url.endsWith("/") ? url : `${url}/`;
}

function absoluteBaseUrl(url) {
  return new URL(normalizeBaseUrl(url), window.location.href).href;
}

// `?data=` is the documented spelling; `data_base` / `dataBaseUrl` are the older names
// and stay accepted so existing deep links keep working.
function queryDataBaseUrl() {
  const parameters = new URLSearchParams(window.location.search);
  return parameters.get("data") || parameters.get("data_base") || parameters.get("dataBaseUrl");
}

function configuredDataBaseUrl() {
  const queryValue = queryDataBaseUrl();
  if (queryValue === "local") return "./data/";
  return (
    window.OCEANBENCH_VIEWER_CONFIG?.dataBaseUrl ||
    queryValue ||
    fileConfig?.dataBaseUrl ||
    DEFAULT_REMOTE_DATA_BASE_URL
  );
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
  return (
    window.OCEANBENCH_VIEWER_CONFIG?.columnsBaseUrl ||
    queryValue ||
    fileConfig?.columnsBaseUrl ||
    dataBaseUrl
  );
}

let dataBaseUrl = absoluteBaseUrl(configuredDataBaseUrl());
let columnsBaseUrl = absoluteBaseUrl(configuredColumnsBaseUrl());

function publishResolvedBase() {
  window.__oceanbenchViewerDataBaseUrl = dataBaseUrl;
}

publishResolvedBase();

export function viewerDataBaseUrl() {
  return dataBaseUrl;
}

export function viewerColumnsBaseUrl() {
  return columnsBaseUrl;
}

/**
 * Load the optional viewer-config.json and recompute the data roots from it.
 * A missing file, a non-JSON body or a network failure all leave the roots exactly as
 * the query parameters and the built-in default resolved them. Await this before the
 * first data fetch; the resolvers below read the live values, so nothing is captured
 * at import time.
 */
export async function initializeViewerConfig() {
  try {
    const response = await fetch(new URL(VIEWER_CONFIG_FILE, window.location.href).href, { cache: "no-cache" });
    if (!response.ok) return;
    const parsed = await response.json();
    if (!parsed || typeof parsed !== "object") return;
    fileConfig = parsed;
  } catch (error) {
    return;
  }
  dataBaseUrl = absoluteBaseUrl(configuredDataBaseUrl());
  columnsBaseUrl = absoluteBaseUrl(configuredColumnsBaseUrl());
  publishResolvedBase();
}

export function resolveViewerDataUrl(url) {
  if (!url || /^(?:[a-z]+:)?\/\//i.test(url) || url.startsWith("data:")) return url;
  const withoutDataPrefix = url.replace(/^\.?\/*data\//, "");
  return new URL(withoutDataPrefix, dataBaseUrl).href;
}

// The column store conventionally sits beside the pyramid as `<slug>.columns.zarr`.
export function resolveColumnStoreUrl(slug) {
  return new URL(`${slug}.columns.zarr`, columnsBaseUrl).href;
}
