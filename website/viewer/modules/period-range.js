// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// The Period control's vocabulary: the values the URL hash carries and the range of start
// dates each one covers. No dependencies, so the shared state can validate a hash value
// without loading the scores reader (modules/period-view.js).

// Same values and URL spelling as the Scores page (website/scores-summary.js).
export const PERIOD_WHOLE_YEAR = "year";
export const PERIOD_CUSTOM = "custom";
export const PERIOD_QUARTERS = Object.freeze([
  { value: "jan-mar", label: "Jan-Mar", from: "01-01", to: "03-31" },
  { value: "apr-jun", label: "Apr-Jun", from: "04-01", to: "06-30" },
  { value: "jul-sep", label: "Jul-Sep", from: "07-01", to: "09-30" },
  { value: "oct-dec", label: "Oct-Dec", from: "10-01", to: "12-31" },
]);
const CUSTOM_PATTERN = /^(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})$/;

/** Whether `value` is a period the URL may carry: whole year, a quarter or "from_to". */
export function isPeriodValue(value) {
  return value === PERIOD_WHOLE_YEAR || PERIOD_QUARTERS.some((quarter) => quarter.value === value) || CUSTOM_PATTERN.test(String(value));
}

/** The custom value for a range, "YYYY-MM-DD_YYYY-MM-DD". */
export function customPeriodValue(from, to) {
  return `${from}_${to}`;
}

/**
 * The range a period value covers in `year`: { from, to, label, custom }, both ends
 * "YYYY-MM-DD" and inclusive, or null for the whole year.
 */
export function periodRange(value, year) {
  const quarter = PERIOD_QUARTERS.find((entry) => entry.value === value);
  if (quarter) return { from: `${year}-${quarter.from}`, to: `${year}-${quarter.to}`, label: `${quarter.label} ${year}`, custom: false };
  const custom = CUSTOM_PATTERN.exec(String(value));
  if (custom) return { from: custom[1], to: custom[2], label: `${custom[1]} to ${custom[2]}`, custom: true };
  return null;
}

/** Whether a start date ("YYYY-MM-DD...") falls inside a range. */
export function startInRange(date, range) {
  const day = String(date).slice(0, 10);
  return day >= range.from && day <= range.to;
}
