// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Canonical Forecast 1 / Forecast 2 identity, shared by panel headers and every
// per-forecast chart/overlay series. Okabe-Ito blue/orange is colorblind-friendly.
export const FORECAST_COLORS = ["#0072B2", "#D55E00"];

export function forecastColor(index) {
  return FORECAST_COLORS[index] || FORECAST_COLORS[0];
}
