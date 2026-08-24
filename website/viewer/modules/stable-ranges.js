// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

/**
 * Lead-stable axis bounds for the diagnostics beside the map.
 *
 * Sliding the lead day is how a forecast's error growth is read, so an axis that
 * rescales on every step hides the very change the slider is being moved to see. Every
 * diagnostic therefore draws inside a frame that is fixed for the current selection
 * (dataset, variable, region, start date, scope, metric) and only moves when that
 * selection changes.
 *
 * Two kinds of diagnostic feed this:
 *   - those whose artifact already holds every lead (RMSE by depth, RMSE by start,
 *     the lead curve, the water column) compute their exact all-lead extent directly
 *     and never need this registry;
 *   - those computed from the lead currently on screen (the live spectrum, the obs-error
 *     ramp, the difference field) cannot know a future lead's extent, so they keep a
 *     GROW-ONLY bound here: the frame widens to fit new data and never shrinks while the
 *     selection is unchanged, which keeps every lead honest (data is padded, never
 *     clipped) and keeps the axis still on the way back down the slider.
 */

let signature = null;
const bounds = new Map();

/**
 * Point the registry at a selection. Every remembered bound is dropped when the
 * selection changes, so a new dataset/variable/region/start starts from its own data
 * instead of inheriting the previous selection's frame.
 */
export function syncStableRanges(nextSignature) {
  if (nextSignature === signature) return;
  signature = nextSignature;
  bounds.clear();
}

/**
 * Forget every bound whose id starts with `prefix`. A diagnostic whose key carries part of
 * its own state (the spectrum's key carries the box) mints a new id each time that state
 * changes, and the bounds under the old ids are unreachable from then on.
 */
export function forgetStableRanges(prefix) {
  for (const id of [...bounds.keys()]) {
    if (id.startsWith(prefix)) bounds.delete(id);
  }
}

/** Grow-only upper bound: the largest `value` seen for `id` under this selection. */
export function stableMax(id, value) {
  const candidate = Number.isFinite(value) && value > 0 ? value : 0;
  const previous = bounds.get(id);
  const next = previous == null ? candidate : Math.max(previous, candidate);
  bounds.set(id, next);
  return next;
}

/**
 * Grow-only interval: the union of every [low, high] seen for `id` under this selection.
 * A non-finite pair leaves the remembered interval untouched (and returns it), so a lead
 * with no data cannot collapse the frame.
 */
export function stableInterval(id, low, high) {
  const previous = bounds.get(id) || null;
  if (!Number.isFinite(low) || !Number.isFinite(high) || !(high >= low)) return previous;
  const next = previous ? [Math.min(previous[0], low), Math.max(previous[1], high)] : [low, high];
  bounds.set(id, next);
  return next;
}
