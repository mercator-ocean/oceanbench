// SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

const INTERCOM_APPLICATION_IDENTIFIER = "nd4ejzt6";
const INTERCOM_API_BASE_URL = "https://api-iam.intercom.io";
const INTERCOM_WIDGET_URL = `https://widget.intercom.io/widget/${INTERCOM_APPLICATION_IDENTIFIER}`;

window.intercomSettings = {
  api_base: INTERCOM_API_BASE_URL,
  app_id: INTERCOM_APPLICATION_IDENTIFIER,
};

function loadIntercomWidget() {
  const script = document.createElement("script");
  script.async = true;
  script.src = INTERCOM_WIDGET_URL;
  document.head.appendChild(script);
}

function installIntercomWidget() {
  if (typeof window.Intercom === "function") {
    window.Intercom("reattach_activator");
    window.Intercom("update", window.intercomSettings);
    return;
  }
  const intercom = (...args) => intercom.c(args);
  intercom.q = [];
  intercom.c = (args) => intercom.q.push(args);
  window.Intercom = intercom;
  if (document.readyState === "complete") loadIntercomWidget();
  else window.addEventListener("load", loadIntercomWidget);
}

installIntercomWidget();
