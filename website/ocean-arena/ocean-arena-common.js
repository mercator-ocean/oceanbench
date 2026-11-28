// SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

const OceanArenaCommon = (() => {
  const { CHALLENGERS, DIMENSIONS, HEXAGON_RESOLUTION, fakeScores } = OceanArenaChallengers;
  const COMPACT_WIDTH = 900;
  const TOOLTIP_WIDTH = 180;
  const NIGHT_SKY_URL = "assets/night-sky.png";
  const ATMOSPHERE_COLOR = "#4fd1ff";

  const isCompact = (container) => container.clientWidth < COMPACT_WIDTH;

  const defaultSelection = () =>
    Object.fromEntries(Object.entries(DIMENSIONS).map(([dimension, options]) => [dimension, options[0].key]));

  const _pillGroupHtml = (dimension, title) =>
    `<section><h2>${title}</h2><div class="ocean-arena-pills" data-dimension="${dimension}"></div></section>`;

  const arenaHtml = (alternativeLinkHtml) => `
    <div class="ocean-arena-globe"></div>
    <header class="ocean-arena-header">
      <h1>Ocean<span>Bench</span> Arena</h1>
      <p>Who forecasts the ocean best? Each hexagon goes to the winning system.</p>
    </header>
    <aside class="ocean-arena-controls">
      ${_pillGroupHtml("variable", "Variable")}
      ${_pillGroupHtml("horizon", "Lead time")}
      ${_pillGroupHtml("depth", "Depth")}
    </aside>
    <aside class="ocean-arena-leaderboard">
      <h2>Ocean conquered</h2>
      <ol></ol>
      <small>Fake scores — preview only</small>
    </aside>
    <div class="ocean-arena-tooltip"></div>
    <nav class="ocean-arena-links">${alternativeLinkHtml}<a href="#all-scores">Detailed scores ↓</a></nav>`;

  const _oceanCells = (land) =>
    h3.getRes0Cells()
      .flatMap((cell) => h3.cellToChildren(cell, HEXAGON_RESOLUTION))
      .map((cell) => ({ cell, center: h3.cellToLatLng(cell) }))
      .filter(({ center: [latitude, longitude] }) => !d3.geoContains(land, [longitude, latitude]));

  const _toPolygon = ({ cell, center }) => ({
    cell,
    center,
    geometry: { type: "Polygon", coordinates: [[...h3.cellToBoundary(cell, true)].reverse()] },
  });

  const loadOcean = async (landUrl) => {
    const topology = await fetch(landUrl).then((response) => response.json());
    const land = topojson.feature(topology, topology.objects.land);
    return { land, polygons: _oceanCells(land).map(_toPolygon) };
  };

  const scoreCells = (polygons, selection) =>
    polygons.map((polygon) => {
      const ranking = fakeScores(selection, ...polygon.center);
      return { ...polygon, ranking, margin: ranking[1].error - ranking[0].error };
    });

  const winnerColor = (cell) => cell.ranking[0].challenger.color;

  const _leaderboard = (cells) =>
    CHALLENGERS.map((challenger) => ({
      challenger,
      share: cells.filter((cell) => cell.ranking[0].challenger === challenger).length / cells.length,
    })).sort((first, second) => second.share - first.share);

  const _leaderboardHtml = (entries) =>
    entries
      .map(
        ({ challenger, share }) => `
        <li>
          <span class="name"><span class="swatch" style="background:${challenger.color}"></span>${challenger.name}</span>
          <span class="share">${(share * 100).toFixed(1)}%</span>
          <span class="bar"><div style="width:${share * 100}%;background:${challenger.color}"></div></span>
        </li>`,
      )
      .join("");

  const _tooltipHtml = (cell) =>
    `<strong>🏆 ${cell.ranking[0].challenger.name}</strong>` +
    cell.ranking
      .map(({ challenger, error }) => `<div class="row"><span style="color:${challenger.color}">${challenger.name}</span><span>${error.toFixed(3)}</span></div>`)
      .join("");

  const showTooltip = (container, cell, { clientX, clientY }) => {
    const tooltip = container.querySelector(".ocean-arena-tooltip");
    const bounds = container.getBoundingClientRect();
    tooltip.style.display = cell ? "block" : "none";
    tooltip.innerHTML = cell ? _tooltipHtml(cell) : "";
    tooltip.style.left = `${Math.min(clientX - bounds.left + 16, bounds.width - TOOLTIP_WIDTH)}px`;
    tooltip.style.top = `${clientY - bounds.top + 16}px`;
  };

  const _renderPills = (container, selection, onSelect) =>
    container.querySelectorAll(".ocean-arena-pills").forEach((group) => {
      const dimension = group.dataset.dimension;
      group.innerHTML = DIMENSIONS[dimension]
        .map(({ key, label }) => `<button type="button" data-key="${key}" class="${selection[dimension] === key ? "active" : ""}">${label}</button>`)
        .join("");
      group.querySelectorAll("button").forEach((button) =>
        button.addEventListener("click", () => onSelect({ ...selection, [dimension]: button.dataset.key })),
      );
    });

  const renderPanels = (container, cells, selection, onSelect) => {
    container.querySelector(".ocean-arena-leaderboard ol").innerHTML = _leaderboardHtml(_leaderboard(cells));
    _renderPills(container, selection, onSelect);
  };

  const createGlobe = (container, rendererConfig) => {
    const element = container.querySelector(".ocean-arena-globe");
    const globe = Globe({ rendererConfig })(element)
      .backgroundImageUrl(NIGHT_SKY_URL)
      .atmosphereColor(ATMOSPHERE_COLOR)
      .atmosphereAltitude(0.25)
      .width(container.clientWidth)
      .height(container.clientHeight)
      .pointOfView({ lat: 10, lng: -30, altitude: isCompact(container) ? 3.4 : 2.3 });
    globe.controls().autoRotate = true;
    globe.controls().autoRotateSpeed = 0.4;
    globe.controls().enableZoom = !isCompact(container);
    new ResizeObserver(() => globe.width(container.clientWidth).height(container.clientHeight)).observe(container);
    return globe;
  };

  const mount = (container, alternativeLinkHtml, start) => {
    container.innerHTML = arenaHtml(alternativeLinkHtml);
    start(container).catch((error) => {
      container.querySelector(".ocean-arena-header p").textContent = `⚠️ ${error.message}`;
    });
  };

  return Object.freeze({
    HEXAGON_RESOLUTION,
    isCompact,
    defaultSelection,
    loadOcean,
    scoreCells,
    winnerColor,
    showTooltip,
    renderPanels,
    createGlobe,
    mount,
  });
})();
