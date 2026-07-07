// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Reusable, dependency-free "?" method-note affordance (contracts.md "transparent
// science"). attachMethodNote(anchorEl, noteId, dynamicFields?) appends a small
// circular "?" button to the anchor and wires a popover that opens on hover AND on
// click/tap, closes on Escape, outside-click, or blur, and never shifts layout (the
// popover is fixed-position, appended to <body>). Theme-aware styling lives in
// styles.css; the copy lives in method-notes.js.

import { METHOD_NOTES, renderEddyParameters } from "./method-notes.js";

let sharedPopover = null;
let openButton = null;
let hideTimer = null;

function ensurePopover() {
  if (sharedPopover) return sharedPopover;
  const popover = document.createElement("div");
  popover.className = "method-popover";
  popover.setAttribute("role", "tooltip");
  popover.hidden = true;
  popover.addEventListener("mouseenter", () => clearTimeout(hideTimer));
  popover.addEventListener("mouseleave", scheduleHide);
  document.body.appendChild(popover);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hidePopover();
  });
  document.addEventListener("pointerdown", (event) => {
    if (!sharedPopover || sharedPopover.hidden) return;
    if (event.target === openButton || (openButton && openButton.contains(event.target))) return;
    if (sharedPopover.contains(event.target)) return;
    hidePopover();
  });
  window.addEventListener("scroll", hidePopover, true);
  sharedPopover = popover;
  return popover;
}

function substitute(body, dynamicFields) {
  return body.replace(/\{(\w+)\}/g, (match, key) => {
    if (!(key in dynamicFields)) return "";
    const value = dynamicFields[key];
    // `params` is pre-rendered HTML (the eddy parameter list); everything else is text.
    return key === "params" ? String(value) : escapeHtml(String(value));
  });
}

function showPopover(button, noteId, dynamicFields) {
  const note = METHOD_NOTES[noteId];
  if (!note) return;
  const popover = ensurePopover();
  clearTimeout(hideTimer);
  openButton = button;
  popover.innerHTML =
    `<div class="method-popover-title">${escapeHtml(note.title || "")}</div>` +
    `<div class="method-popover-body">${substitute(note.body || "", dynamicFields || {})}</div>`;
  popover.hidden = false;
  button.setAttribute("aria-expanded", "true");
  positionPopover(popover, button);
}

function positionPopover(popover, button) {
  const rect = button.getBoundingClientRect();
  const margin = 8;
  const width = Math.min(320, window.innerWidth - 2 * margin);
  popover.style.width = `${width}px`;
  // Measure after width is set, then clamp inside the viewport (prefer below-right).
  const height = popover.offsetHeight;
  let left = rect.left;
  if (left + width > window.innerWidth - margin) left = window.innerWidth - margin - width;
  left = Math.max(margin, left);
  let top = rect.bottom + 6;
  if (top + height > window.innerHeight - margin) top = Math.max(margin, rect.top - height - 6);
  popover.style.left = `${Math.round(left)}px`;
  popover.style.top = `${Math.round(top)}px`;
}

function scheduleHide() {
  clearTimeout(hideTimer);
  hideTimer = setTimeout(hidePopover, 160);
}

function hidePopover() {
  clearTimeout(hideTimer);
  if (sharedPopover) sharedPopover.hidden = true;
  if (openButton) openButton.setAttribute("aria-expanded", "false");
  openButton = null;
}

/**
 * Attach a "?" method-note button to `anchorEl` for the note `noteId`. Idempotent —
 * an existing button previously attached for the same note is replaced, so re-renders
 * do not accumulate buttons. `dynamicFields` fills `{token}` placeholders in the note
 * body (values are HTML-escaped, except a `params` field which is treated as raw HTML).
 */
export function attachMethodNote(anchorEl, noteId, dynamicFields = {}) {
  if (!anchorEl || !METHOD_NOTES[noteId]) return null;
  const existing = anchorEl.querySelector(`:scope > .method-note-btn[data-note="${noteId}"]`);
  if (existing) existing.remove();
  const button = document.createElement("button");
  button.type = "button";
  button.className = "method-note-btn";
  button.dataset.note = noteId;
  button.textContent = "?";
  button.setAttribute("aria-expanded", "false");
  button.setAttribute("aria-label", `Method note: ${METHOD_NOTES[noteId].title}`);
  button.addEventListener("mouseenter", () => showPopover(button, noteId, dynamicFields));
  button.addEventListener("mouseleave", scheduleHide);
  button.addEventListener("focus", () => showPopover(button, noteId, dynamicFields));
  button.addEventListener("blur", hidePopover);
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (button.getAttribute("aria-expanded") === "true") hidePopover();
    else showPopover(button, noteId, dynamicFields);
  });
  anchorEl.appendChild(button);
  return button;
}

// Convenience wrapper used by the eddy legend: renders the live census parameters block
// into the note's {params} token, or falls back to the fixed text alone.
export function attachEddyMethodNote(anchorEl, parameters) {
  return attachMethodNote(anchorEl, "eddies-legend", { params: renderEddyParameters(parameters) });
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (character) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]),
  );
}
