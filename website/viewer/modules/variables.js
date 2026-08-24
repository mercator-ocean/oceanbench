// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// The viewer's variable vocabulary: which keys name a current, what depth a key means,
// and what a manifest entry looks like for the synthetic currents channels the pyramids
// do not carry. Pure functions of a manifest and a variable key, so they hold no view
// state and can be read without reading app.js.
//
// The speed colormap and its top-of-scale value live here because the synthetic currents
// entry has to declare them; modules/particles.js names the same colormap for the flow
// lines it draws over the same bar.

export const SPEED_COLORMAP = "speed";
export const CURRENTS_MAX_SPEED = 1.2; // m/s mapping to the top of the speed colormap

// Currents are a synthetic variable (speed magnitude √(u²+v²)) built from the u/v
// velocity components, one per available depth, so they sit in the variable dropdown
// like any other channel. The particle animation is an optional overlay on top.
export const CURRENTS_VARIABLE_SURFACE = "current_speed";
export const CURRENTS_VARIABLE_15M = "current_speed_15m";

export function isCurrentsVariable(key) {
  return key === CURRENTS_VARIABLE_SURFACE || key === CURRENTS_VARIABLE_15M;
}

export function currentsVariableDepth(key) {
  return key === CURRENTS_VARIABLE_15M ? "15m" : "surface";
}

// Human-facing depth label ("15 m" with a thin space), separate from the "15m" token
// used for depth-bin logic elsewhere.
export function currentsDepthLabel(key) {
  return key === CURRENTS_VARIABLE_15M ? "15 m" : "surface";
}

// Class-4 current observations are surface drifters drogued at 15 m: obs and skill for
// velocities exist ONLY at the "15m" depth. A surface current selection (surface u, v,
// or derived surface current_speed) therefore has no honest obs to compare against.
export function isVelocityFamilyVariable(key) {
  return isCurrentsVariable(key) || String(key).includes("sea_water_velocity");
}

export function isSurfaceCurrentVariable(key) {
  return isVelocityFamilyVariable(key) && !String(key).endsWith("_15m");
}

// Matching 15 m variable for a surface current selection (u→u_15m, current_speed→…_15m).
export function matching15mCurrentVariable(key) {
  return isSurfaceCurrentVariable(key) ? `${key}_15m` : key;
}

export function syntheticCurrentsEntry(key) {
  return {
    standard_name: "sea_water_speed",
    units: "m/s",
    depth: currentsVariableDepth(key),
    default_colormap: SPEED_COLORMAP,
    default_range: [0, CURRENTS_MAX_SPEED],
  };
}

// Real manifest entry, or a synthetic descriptor for the currents variables.
export function variableEntry(manifest, key) {
  if (isCurrentsVariable(key)) return syntheticCurrentsEntry(key);
  return manifest && manifest.variables[key];
}

export function variableExists(manifest, key) {
  if (isCurrentsVariable(key)) return currentsVariableOptions(manifest).some((option) => option.value === key);
  return Boolean(manifest && key in manifest.variables);
}

// Currents variable options available for this manifest, gated on the u/v components.
export function currentsVariableOptions(manifest) {
  if (!manifest || !manifest.variables) return [];
  const options = [];
  if ("eastward_sea_water_velocity" in manifest.variables && "northward_sea_water_velocity" in manifest.variables) {
    options.push({ value: CURRENTS_VARIABLE_SURFACE, label: "Currents · surface" });
  }
  if ("eastward_sea_water_velocity_15m" in manifest.variables && "northward_sea_water_velocity_15m" in manifest.variables) {
    options.push({ value: CURRENTS_VARIABLE_15M, label: "Currents · 15 m" });
  }
  return options;
}

export function prettyName(standardName) {
  return standardName.replace(/_/g, " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

export function variableLabel(manifest, key) {
  const entry = manifest.variables[key];
  return `${prettyName(entry.standard_name)} · ${entry.depth}`;
}
