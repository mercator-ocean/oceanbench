// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Filling a <select> without disturbing it. Rebuilding the option list on every refresh
// closes an open dropdown and loses the keyboard position, so the list is only replaced
// when it actually differs from what the element already holds.

export function selectAlreadyHolds(select, options) {
  if (select.options.length !== options.length) return false;
  for (let i = 0; i < options.length; i += 1) {
    if (select.options[i].value !== String(options[i].value)) return false;
    if (select.options[i].textContent !== options[i].label) return false;
  }
  return true;
}

// Rewriting a select destroys and rebuilds its option nodes, which closes an open
// dropdown under the user's pointer. renderPanel refreshes the panel controls on every
// paint, so while the leads play this fired several times a second and a picker could
// not be held open long enough to choose from. Write only what actually differs.
export function populateSelect(select, options, selectedValue) {
  if (selectAlreadyHolds(select, options)) {
    if (String(select.value) !== String(selectedValue)) select.value = String(selectedValue);
    return;
  }
  select.innerHTML = "";
  for (const option of options) {
    const element = document.createElement("option");
    element.value = option.value;
    element.textContent = option.label;
    if (String(option.value) === String(selectedValue)) element.selected = true;
    select.appendChild(element);
  }
}
