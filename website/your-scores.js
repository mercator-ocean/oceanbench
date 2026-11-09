// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

(function (root) {
  const SUPPORTED_SCHEMA_VERSION = 1;
  const YOUR_SCORES_KEY_PREFIX = "your__";
  const UNSAFE_KEY_CHARACTERS = /[^a-z0-9_]+/g;
  const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;
  const DEPTH_METRIC_KEYS = new Set(["rmsd_variables_glorys", "rmsd_variables_glo12", "rmsd_variables_observations"]);
  const FLAT_DEPTH_KEY = "flat";

  const METRIC_KEYS = {
    "rmsd_of_variables:glorys_reanalysis": "rmsd_variables_glorys",
    "rmsd_of_mixed_layer_depth:glorys_reanalysis": "rmsd_mld_glorys",
    "rmsd_of_geostrophic_currents:glorys_reanalysis": "rmsd_geostrophic_glorys",
    "deviation_of_lagrangian_trajectories:glorys_reanalysis": "lagrangian_glorys",
    "rmsd_of_variables:glo12_analysis": "rmsd_variables_glo12",
    "rmsd_of_mixed_layer_depth:glo12_analysis": "rmsd_mld_glo12",
    "rmsd_of_geostrophic_currents:glo12_analysis": "rmsd_geostrophic_glo12",
    "deviation_of_lagrangian_trajectories:glo12_analysis": "lagrangian_glo12",
    "rmsd_of_variables:observations": "rmsd_variables_observations",
  };

  const isObject = (value) => typeof value === "object" && value !== null && !Array.isArray(value);
  const isNumberOrNull = (value) => value === null || (typeof value === "number" && Number.isFinite(value));
  const isIntegerAtLeast = (minimum) => (value) => Number.isInteger(value) && value >= minimum;
  const isNonEmptyString = (value) => typeof value === "string" && value.length > 0;

  const REQUIRED_ROW_CHECKS = {
    metric: (value) => Object.keys(METRIC_KEYS).some((key) => key.startsWith(`${value}:`)),
    reference: (value) => Object.keys(METRIC_KEYS).some((key) => key.endsWith(`:${value}`)),
    variable: isNonEmptyString,
    standard_name: (value) => value === null || typeof value === "string",
    depth: isNonEmptyString,
    unit: (value) => typeof value === "string",
    lead_day: isIntegerAtLeast(1),
    value: isNumberOrNull,
  };

  function rowErrors(row, index) {
    if (!isObject(row)) return [`rows[${index}] is not an object`];
    return Object.entries(REQUIRED_ROW_CHECKS)
      .filter(([field, isValid]) => !(field in row) || !isValid(row[field]))
      .map(([field]) => `rows[${index}].${field} is missing or invalid`);
  }

  function headerErrors(document) {
    const checks = {
      oceanbench_version: isNonEmptyString,
      challenger: isNonEmptyString,
      region: (value) => isObject(value) && isNonEmptyString(value.id),
      start_dates: (value) => Array.isArray(value) && value.length > 0 && value.every((date) => DATE_PATTERN.test(date)),
      lead_days_count: isIntegerAtLeast(1),
      challenger_code: (value) => typeof value === "string",
      created_at: isNonEmptyString,
      timings: isObject,
      rows: Array.isArray,
    };
    return Object.entries(checks)
      .filter(([field, isValid]) => !(field in document) || !isValid(document[field]))
      .map(([field]) => `${field} is missing or invalid`);
  }

  /** The reasons a parsed JSON value is not a supported scores file; empty when it is one. */
  function scoresFileErrors(document) {
    if (!isObject(document)) return ["The file is not a JSON object."];
    if (document.schema_version !== SUPPORTED_SCHEMA_VERSION) {
      return [
        `schema_version is ${document.schema_version === undefined ? "missing" : JSON.stringify(document.schema_version)}, ` +
          `this page reads version ${SUPPORTED_SCHEMA_VERSION}. Evaluate again with a recent oceanbench release`,
      ];
    }
    const errors = headerErrors(document);
    return errors.length > 0 ? errors : document.rows.flatMap(rowErrors).slice(0, 5);
  }

  function capitalize(text) {
    return text.charAt(0).toUpperCase() + text.slice(1).toLowerCase();
  }

  function depthKey(metricKey, depth) {
    return DEPTH_METRIC_KEYS.has(metricKey) ? capitalize(depth) : FLAT_DEPTH_KEY;
  }

  function addRow(depthsByMetric, row) {
    const metricKey = METRIC_KEYS[`${row.metric}:${row.reference}`];
    const depths = depthsByMetric[metricKey] || {};
    const depth = depths[depthKey(metricKey, row.depth)] || { variables: {} };
    const variable = depth.variables[row.variable] || {
      standard_name: row.standard_name || "",
      unit: row.unit,
      data: {},
    };
    return {
      ...depthsByMetric,
      [metricKey]: {
        ...depths,
        [depthKey(metricKey, row.depth)]: {
          variables: {
            ...depth.variables,
            [row.variable]: { ...variable, data: { ...variable.data, [String(row.lead_day)]: row.value } },
          },
        },
      },
    };
  }

  /** The scores of a valid scores file, in the shape of the published scores of the page. */
  function scoresFileToModelScores(document, name) {
    const depthsByMetric = document.rows.reduce(addRow, {});
    return Object.fromEntries(
      Object.entries(depthsByMetric).map(([metricKey, depths]) => [metricKey, { name, depths }]),
    );
  }

  /** A key for a table row that is safe to put in HTML and cannot collide with a published challenger. */
  function yourScoresKey(challengerName) {
    return YOUR_SCORES_KEY_PREFIX + challengerName.toLowerCase().replace(UNSAFE_KEY_CHARACTERS, "_");
  }

  const api = { scoresFileErrors, scoresFileToModelScores, yourScoresKey, SUPPORTED_SCHEMA_VERSION };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.OceanBenchYourScores = api;
})(typeof window !== "undefined" ? window : globalThis);
