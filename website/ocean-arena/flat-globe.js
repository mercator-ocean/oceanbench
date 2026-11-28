// SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

(() => {
  const LAND_URL = "assets/land-110m.json";
  const TEXTURE_WIDTH = 4096;
  const TEXTURE_HEIGHT = 2048;
  const MARGIN_FOR_FULL_OPACITY = 0.4;
  const OCEAN_COLOR = "#02060f";
  const LAND_COLOR = "#1a2438";
  const ALTERNATIVE_LINK_HTML = '<a href="ocean-arena-3d.html">3D version</a>';
  const arena = OceanArenaCommon;

  const _createPainter = (land) => {
    const canvas = Object.assign(document.createElement("canvas"), { width: TEXTURE_WIDTH, height: TEXTURE_HEIGHT });
    const context = canvas.getContext("2d");
    const projection = d3.geoEquirectangular().fitSize([TEXTURE_WIDTH, TEXTURE_HEIGHT], { type: "Sphere" });
    const path = d3.geoPath(projection, context);
    const fillPath = (geometry, color) => {
      context.beginPath();
      path(geometry);
      context.fillStyle = color;
      context.fill();
    };
    const paintCell = (cell) => {
      context.globalAlpha = 0.45 + 0.55 * Math.min(cell.margin / MARGIN_FOR_FULL_OPACITY, 1);
      fillPath(cell.geometry, arena.winnerColor(cell));
      context.globalAlpha = 1;
      context.stroke();
    };
    return (cells) => {
      context.fillStyle = OCEAN_COLOR;
      context.fillRect(0, 0, TEXTURE_WIDTH, TEXTURE_HEIGHT);
      context.strokeStyle = OCEAN_COLOR;
      context.lineWidth = 3;
      cells.forEach(paintCell);
      fillPath(land, LAND_COLOR);
      return canvas.toDataURL("image/jpeg", 0.9);
    };
  };

  const _render = (container, globe, polygons, paint, cellsByIdentifier) => (selection) => {
    const cells = arena.scoreCells(polygons, selection);
    cellsByIdentifier.splice(0, 1, new Map(cells.map((cell) => [cell.cell, cell])));
    globe.globeImageUrl(paint(cells));
    arena.renderPanels(container, cells, selection, _render(container, globe, polygons, paint, cellsByIdentifier));
  };

  const _start = async (container) => {
    const { land, polygons } = await arena.loadOcean(LAND_URL);
    const compact = arena.isCompact(container);
    const globe = arena.createGlobe(container, { antialias: !compact, powerPreference: "high-performance" });
    const cellsByIdentifier = [new Map()];
    globe.renderer().setPixelRatio(Math.min(window.devicePixelRatio, compact ? 1.5 : 2));
    globe.onGlobeClick(({ lat, lng }, event) => {
      globe.controls().autoRotate = false;
      arena.showTooltip(container, cellsByIdentifier[0].get(h3.latLngToCell(lat, lng, arena.HEXAGON_RESOLUTION)), event);
    });
    _render(container, globe, polygons, _createPainter(land), cellsByIdentifier)(arena.defaultSelection());
  };

  document.querySelectorAll(".ocean-arena[data-version='flat']").forEach((container) =>
    arena.mount(container, ALTERNATIVE_LINK_HTML, _start),
  );
})();
