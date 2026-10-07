// SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

const EXTERNAL_LINK_CLASS = "external-link";
const GRAPHIC_SELECTOR = "img, i, svg";

function isExternalTextLink(link) {
  return (
    link.hostname !== "" &&
    link.hostname !== window.location.hostname &&
    !link.querySelector(GRAPHIC_SELECTOR)
  );
}

function markExternalLinks() {
  Array.from(document.querySelectorAll(`a[href]:not(.${EXTERNAL_LINK_CLASS})`))
    .filter(isExternalTextLink)
    .forEach((link) => link.classList.add(EXTERNAL_LINK_CLASS));
}

markExternalLinks();
new MutationObserver(markExternalLinks).observe(document.body, { childList: true, subtree: true });
