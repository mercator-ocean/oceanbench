// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2
//
// Classic (non-module) renderer for the local-evaluation overlay scorecard. Reads its data from
// an inlined <script type="application/json" id="scorecard-data"> element rather than fetch(), so
// the report opens over file:// with no server. Same no-ranking scorecard semantics as the website
// scores page (mean +/- 95% CI over forecast starts, baselines pinned, neutral order), with the
// user's model overlaid on the published challengers and highlighted.
(function () {
  var VARIABLE_LABELS = {
    sea_surface_height_above_geoid: "sea surface height",
    sea_water_potential_temperature: "temperature",
    sea_water_salinity: "salinity",
    northward_sea_water_velocity: "meridional current",
    eastward_sea_water_velocity: "zonal current",
    ocean_mixed_layer_thickness: "mixed layer depth",
    geostrophic_northward_sea_water_velocity: "meridional geostrophic current",
    geostrophic_eastward_sea_water_velocity: "zonal geostrophic current"
  };
  var REFERENCE_LABELS = { glorys: "GLORYS reanalysis", glo12: "GLO12 analysis", observations: "observations (Class-4)" };
  var METRIC_PHRASE = { rmsd: "RMSD", class4_rmsd: "Class-4 RMSD" };
  var DEPTH_ORDER = ["surface", "50m", "100m", "200m", "300m", "500m", "0-5m", "5-100m", "15m", "100-300m", "300-600m"];
  var NULL_KEY = "∅";

  var payload = JSON.parse(document.getElementById("scorecard-data").textContent);
  var cells = payload.cells;
  var challengers = payload.challengers;

  function nullable(v) { return (v === null || v === undefined || v === "") ? null : v; }
  function distinct(a) { return Array.prototype.filter.call(a, function (v, i) { return a.indexOf(v) === i; }); }
  function variableLabel(v) { return VARIABLE_LABELS[v] || v || "-"; }
  function challengerLabel(s) { return (challengers[s] && challengers[s].display_name) || s; }
  function isBaseline(s) { return !!(challengers[s] && challengers[s].is_baseline); }
  function isYourModel(s) { return !!(challengers[s] && challengers[s].is_your_model); }
  function metricForReference(r) { return r === "observations" ? "class4_rmsd" : "rmsd"; }
  function sortedDepths(ds) {
    return ds.slice().sort(function (a, b) {
      var ia = DEPTH_ORDER.indexOf(a), ib = DEPTH_ORDER.indexOf(b);
      return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
    });
  }
  function fmt(v) {
    if (v === null || v === undefined || !isFinite(v)) return "-";
    var m = Math.abs(v);
    if (m >= 100) return v.toFixed(1);
    if (m >= 1) return v.toFixed(2);
    return v.toFixed(3);
  }

  var selection = { reference: null, depth: null, lead: null };

  function relevantCells() {
    return cells.filter(function (c) { return c.region === payload.region && Number(c.year) === Number(payload.year); });
  }

  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  function buildControls() {
    var rel = relevantCells();
    var references = distinct(rel.map(function (c) { return nullable(c.reference); }).filter(Boolean)).sort();
    selection.reference = references.indexOf("glorys") >= 0 ? "glorys" : references[0];
    var leads = distinct(rel.map(function (c) { return Number(c.lead_day); })).sort(function (a, b) { return a - b; });
    selection.lead = leads.indexOf(5) >= 0 ? 5 : leads[Math.min(2, leads.length - 1)];

    var controls = document.getElementById("controls");
    controls.innerHTML = "";
    controls.appendChild(labelledSelect("Reference", "reference-select", references, selection.reference,
      function (r) { return REFERENCE_LABELS[r] || r; }));
    controls.appendChild(labelledSelect("Depth", "depth-select", [], null, null));
    controls.appendChild(labelledSelect("Lead day", "lead-select", leads, selection.lead, function (l) { return "day " + l; }));

    document.getElementById("reference-select").addEventListener("change", function (e) {
      selection.reference = e.target.value; refreshDepth(); render();
    });
    document.getElementById("depth-select").addEventListener("change", function (e) { selection.depth = e.target.value; render(); });
    document.getElementById("lead-select").addEventListener("change", function (e) { selection.lead = Number(e.target.value); render(); });
    refreshDepth();
  }

  function labelledSelect(labelText, id, options, selected, labelFor) {
    var label = el("label", null, labelText);
    var select = el("select", { id: id });
    populate(select, options, selected, labelFor);
    label.appendChild(select);
    return label;
  }

  function populate(select, options, selected, labelFor) {
    select.innerHTML = "";
    options.forEach(function (o) {
      var opt = el("option", { value: String(o) }, labelFor ? labelFor(o) : String(o));
      if (String(o) === String(selected)) opt.selected = true;
      select.appendChild(opt);
    });
  }

  function refreshDepth() {
    var control = document.getElementById("depth-select").parentNode;
    if (metricForReference(selection.reference) === "class4_rmsd") { control.hidden = true; selection.depth = null; return; }
    control.hidden = false;
    var depths = sortedDepths(distinct(relevantCells().filter(function (c) {
      return c.metric === "rmsd" && c.reference === selection.reference && c.depth;
    }).map(function (c) { return c.depth; })));
    if (depths.indexOf(selection.depth) < 0) selection.depth = depths[0] || null;
    populate(document.getElementById("depth-select"), depths, selection.depth);
  }

  function columns() {
    var metric = metricForReference(selection.reference);
    var relevant = relevantCells().filter(function (c) {
      return c.metric === metric && Number(c.lead_day) === selection.lead &&
        (metric === "class4_rmsd" ? true : (c.reference === selection.reference && (c.depth === null || c.depth === selection.depth)));
    });
    var seen = {};
    var out = [];
    relevant.forEach(function (c) {
      var k = (c.variable || NULL_KEY) + "|" + (c.depth || NULL_KEY);
      if (!seen[k]) { seen[k] = { variable: c.variable, depth: c.depth, unit: c.unit }; out.push(seen[k]); }
    });
    out.sort(function (a, b) { return variableLabel(a.variable).localeCompare(variableLabel(b.variable)); });
    return out;
  }

  function cellFor(slug, column) {
    var metric = metricForReference(selection.reference);
    var reference = metric === "class4_rmsd" ? "observations" : selection.reference;
    for (var i = 0; i < cells.length; i++) {
      var c = cells[i];
      if (c.challenger === slug && c.region === payload.region && Number(c.year) === Number(payload.year) &&
        c.metric === metric && nullable(c.reference) === reference &&
        nullable(c.variable) === (column.variable || null) && nullable(c.depth) === (column.depth || null) &&
        Number(c.lead_day) === selection.lead) return c;
    }
    return null;
  }

  function orderedSlugs() {
    var slugs = distinct(relevantCells().map(function (c) { return c.challenger; }));
    var your = slugs.filter(isYourModel);
    var baselines = slugs.filter(function (s) { return isBaseline(s) && !isYourModel(s); })
      .sort(function (a, b) { return challengerLabel(a).localeCompare(challengerLabel(b)); });
    var others = slugs.filter(function (s) { return !isBaseline(s) && !isYourModel(s); })
      .sort(function (a, b) { return challengerLabel(a).localeCompare(challengerLabel(b)); });
    return your.concat(baselines, others);
  }

  function render() {
    var cols = columns();
    var metric = metricForReference(selection.reference);
    var depthText = metric === "class4_rmsd" ? "" : (" at " + selection.depth);
    document.getElementById("scorecard-note").textContent =
      METRIC_PHRASE[metric] + " vs " + (REFERENCE_LABELS[selection.reference] || selection.reference) +
      ", " + payload.region + ", day " + selection.lead + depthText + ". Mean ± 95% CI over forecast starts.";

    var thead = document.querySelector("#scorecard thead");
    var headRow = el("tr");
    headRow.appendChild(el("th", null, "model"));
    cols.forEach(function (col) {
      var th = el("th");
      th.appendChild(el("span", null, variableLabel(col.variable) + (col.depth && col.depth !== "surface" ? " @ " + col.depth : "")));
      if (col.unit) th.appendChild(el("span", { "class": "unit" }, col.unit));
      headRow.appendChild(th);
    });
    thead.innerHTML = "";
    thead.appendChild(headRow);

    var tbody = document.querySelector("#scorecard tbody");
    tbody.innerHTML = "";
    orderedSlugs().forEach(function (slug) {
      var tr = el("tr");
      if (isYourModel(slug)) tr.className = "your-model";
      else if (isBaseline(slug)) tr.className = "baseline";
      tr.appendChild(el("td", null, challengerLabel(slug)));
      cols.forEach(function (col) {
        var td = el("td");
        var c = cellFor(slug, col);
        if (!c || c.mean === null || c.mean === undefined) {
          td.appendChild(el("span", { "class": "cell-empty" }, "-"));
          tr.appendChild(td);
          return;
        }
        td.appendChild(el("span", { "class": "cell-mean" }, fmt(c.mean)));
        if (isFinite(c.ci_low) && isFinite(c.ci_high)) {
          td.appendChild(el("span", { "class": "cell-ci" }, "± " + fmt((c.ci_high - c.ci_low) / 2)));
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }

  document.getElementById("provenance").textContent =
    "Generated " + payload.generated_at + " · region " + payload.region + " · year " + payload.year + ".";

  if (!cells.length) {
    var status = document.getElementById("status");
    status.className = "status error";
    status.textContent = "No scorecard data.";
  } else {
    buildControls();
    render();
    document.getElementById("status").hidden = true;
    document.getElementById("main").hidden = false;
  }
})();
