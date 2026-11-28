// SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

(() => {
  const LAND_URL = "assets/land-110m.json";
  const EARTH_URL = "assets/earth-dark.jpg";
  const MAXIMUM_ALTITUDE = 0.06;
  const MARGIN_FOR_MAXIMUM_ALTITUDE = 0.4;
  const TRANSITION_MILLISECONDS = 600;
  const HIGHLIGHT_COLOR = "#ffffff";
  const ALTERNATIVE_LINK_HTML = '<a href="./">Fast version</a>';
  const arena = OceanArenaCommon;

  const _altitude = (cell) => 0.005 + (Math.min(cell.margin, MARGIN_FOR_MAXIMUM_ALTITUDE) / MARGIN_FOR_MAXIMUM_ALTITUDE) * MAXIMUM_ALTITUDE;

  const _capColor = (highlightedCell) => (cell) => (cell === highlightedCell ? HIGHLIGHT_COLOR : arena.winnerColor(cell));

  const _configurePolygons = (globe) =>
    globe
      .globeImageUrl(EARTH_URL)
      .polygonGeoJsonGeometry("geometry")
      .polygonCapColor(_capColor(null))
      .polygonSideColor((cell) => `${arena.winnerColor(cell)}66`)
      .polygonStrokeColor(() => "rgba(2, 6, 15, 0.6)")
      .polygonAltitude(_altitude)
      .polygonsTransitionDuration(TRANSITION_MILLISECONDS);

  const _bindHover = (container, globe) => {
    const lastPointer = [{ clientX: 0, clientY: 0 }];
    container.addEventListener("pointermove", (event) => lastPointer.splice(0, 1, event));
    globe.onPolygonHover((cell) => {
      arena.showTooltip(container, cell, lastPointer[0]);
      globe.polygonCapColor(_capColor(cell));
    });
  };

  const _render = (container, globe, polygons) => (selection) => {
    const cells = arena.scoreCells(polygons, selection);
    globe.polygonsData(cells);
    arena.renderPanels(container, cells, selection, _render(container, globe, polygons));
  };

  const _start = async (container) => {
    const { polygons } = await arena.loadOcean(LAND_URL);
    const globe = _configurePolygons(arena.createGlobe(container, { antialias: true }));
    _bindHover(container, globe);
    _render(container, globe, polygons)(arena.defaultSelection());
  };

  document.querySelectorAll(".ocean-arena[data-version='raised']").forEach((container) =>
    arena.mount(container, ALTERNATIVE_LINK_HTML, _start),
  );
})();
