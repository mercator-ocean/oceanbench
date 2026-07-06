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
