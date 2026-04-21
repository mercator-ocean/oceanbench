// SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

(function () {
  "use strict";

  const LAND_TOPOLOGY_URL = "assets/land-110m.json";
  const WIDTH = 640;
  const HEIGHT = 220;
  const MARGIN = 16;
  const GLOBE_CENTER_X_RATIO = 0.5;
  const DRAG_SENSITIVITY = 0.35;
  const MINIMUM_ZOOM = 0.85;
  const MAXIMUM_ZOOM = 8;
  const ZOOM_SENSITIVITY = 0.0015;
  const SVG_NAMESPACE = "http://www.w3.org/2000/svg";

  const state = {
    activeRegion: null,
    animationFrame: null,
    current: null,
    dragging: false,
    landFeature: null,
    landPromise: null,
    options: null,
    rotation: null,
    zoom: 1,
  };

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  function shortestAngleDelta(start, end) {
    return ((((end - start) % 360) + 540) % 360) - 180;
  }

  function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
  }

  function loadLandFeature() {
    if (state.landFeature) return Promise.resolve(state.landFeature);
    if (!state.landPromise) {
      state.landPromise = fetch(LAND_TOPOLOGY_URL)
        .then((response) => {
          if (!response.ok) throw new Error(`Failed to load ${LAND_TOPOLOGY_URL}`);
          return response.json();
        })
        .then((topology) => {
          state.landFeature = window.topojson.feature(topology, topology.objects.land);
          return state.landFeature;
        })
        .catch(() => null);
    }
    return state.landPromise;
  }

  function regionCenter(regionId, regionMetadata) {
    const bounds = regionMetadata[regionId]?.bounds;
    if (!bounds) return [0, 12];
    return [
      (bounds.minimum_longitude + bounds.maximum_longitude) / 2,
      (bounds.minimum_latitude + bounds.maximum_latitude) / 2,
    ];
  }

  function targetRotation(regionId, regionMetadata) {
    if (!regionMetadata[regionId]?.bounds) {
      return [180, 0, 0];
    }
    const [longitude, latitude] = regionCenter(regionId, regionMetadata);
    return [-longitude, -latitude, 0];
  }

  function targetZoom(regionId, regionMetadata) {
    const bounds = regionMetadata[regionId]?.bounds;
    if (!bounds) return 1;

    const longitudeSpan = Math.abs(bounds.maximum_longitude - bounds.minimum_longitude);
    const latitudeSpan = Math.abs(bounds.maximum_latitude - bounds.minimum_latitude);
    const span = Math.max(longitudeSpan, latitudeSpan, 1);
    return clamp(90 / span, 1.4, 5);
  }

  function edgePoints(start, end, count) {
    const points = [];
    for (let index = 0; index <= count; index += 1) {
      const ratio = index / count;
      points.push([
        start[0] + (end[0] - start[0]) * ratio,
        start[1] + (end[1] - start[1]) * ratio,
      ]);
    }
    return points;
  }

  function regionFillPolygon(regionId, regionMetadata) {
    const bounds = regionMetadata[regionId]?.bounds;
    if (!bounds) return null;

    const minimumLongitude = bounds.minimum_longitude;
    const maximumLongitude = bounds.maximum_longitude;
    const minimumLatitude = bounds.minimum_latitude;
    const maximumLatitude = bounds.maximum_latitude;
    const edgeCount = 18;

    const coordinates = [
      ...edgePoints([minimumLongitude, minimumLatitude], [minimumLongitude, maximumLatitude], edgeCount),
      ...edgePoints([minimumLongitude, maximumLatitude], [maximumLongitude, maximumLatitude], edgeCount).slice(1),
      ...edgePoints([maximumLongitude, maximumLatitude], [maximumLongitude, minimumLatitude], edgeCount).slice(1),
      ...edgePoints([maximumLongitude, minimumLatitude], [minimumLongitude, minimumLatitude], edgeCount).slice(1),
    ];

    return {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [coordinates],
      },
    };
  }

  function regionOutline(regionId, regionMetadata) {
    const bounds = regionMetadata[regionId]?.bounds;
    if (!bounds) return null;

    const minimumLongitude = bounds.minimum_longitude;
    const maximumLongitude = bounds.maximum_longitude;
    const minimumLatitude = bounds.minimum_latitude;
    const maximumLatitude = bounds.maximum_latitude;
    const edgeCount = 18;

    return {
      type: "Feature",
      geometry: {
        type: "MultiLineString",
        coordinates: [
          edgePoints([minimumLongitude, minimumLatitude], [maximumLongitude, minimumLatitude], edgeCount),
          edgePoints([maximumLongitude, minimumLatitude], [maximumLongitude, maximumLatitude], edgeCount),
          edgePoints([maximumLongitude, maximumLatitude], [minimumLongitude, maximumLatitude], edgeCount),
          edgePoints([minimumLongitude, maximumLatitude], [minimumLongitude, minimumLatitude], edgeCount),
        ],
      },
    };
  }

  function activeRegionLabel() {
    const { activeRegion, regionLabels } = state.options;
    return regionLabels[activeRegion] || activeRegion;
  }

  function setStatus(message) {
    const container = document.getElementById("region-globe");
    if (!container) return;
    container.innerHTML = `<div class="region-globe-status">${message}</div>`;
  }

  function updatePaths() {
    if (!state.current) return;
    const { baseScale, projection, path, svg } = state.current;
    projection.rotate(state.rotation);
    projection.scale(baseScale * state.zoom);
    svg.querySelectorAll(".region-globe-path").forEach((element) => {
      element.setAttribute("d", path(element.__data__) || "");
    });
  }

  function appendSvgElement(parent, tagName, attributes = {}, data = null) {
    const element = document.createElementNS(SVG_NAMESPACE, tagName);
    for (const [name, value] of Object.entries(attributes)) {
      element.setAttribute(name, value);
    }
    if (data !== null) {
      element.__data__ = data;
    }
    parent.appendChild(element);
    return element;
  }

  function appendPath(parent, data, className) {
    return appendSvgElement(parent, "path", { class: `region-globe-path ${className}` }, data);
  }

  function stopAnimation() {
    if (state.animationFrame !== null) {
      cancelAnimationFrame(state.animationFrame);
      state.animationFrame = null;
    }
  }

  function animateTo(target, zoom) {
    if (!state.rotation) {
      state.rotation = target;
      state.zoom = zoom;
      updatePaths();
      return;
    }

    stopAnimation();
    const start = state.rotation.slice();
    const startZoom = state.zoom;
    const longitudeDelta = shortestAngleDelta(start[0], target[0]);
    const latitudeDelta = target[1] - start[1];
    const zoomDelta = zoom - startZoom;
    const duration = 450;
    const startTime = performance.now();

    function step(now) {
      const ratio = clamp((now - startTime) / duration, 0, 1);
      const eased = easeOutCubic(ratio);
      state.rotation = [
        start[0] + longitudeDelta * eased,
        start[1] + latitudeDelta * eased,
        0,
      ];
      state.zoom = startZoom + zoomDelta * eased;
      updatePaths();
      if (ratio < 1) {
        state.animationFrame = requestAnimationFrame(step);
      } else {
        state.animationFrame = null;
      }
    }

    state.animationFrame = requestAnimationFrame(step);
  }

  function attachDrag(svgNode) {
    let startX = 0;
    let startY = 0;
    let startRotation = null;

    svgNode.addEventListener("pointerdown", (event) => {
      state.dragging = true;
      stopAnimation();
      startX = event.clientX;
      startY = event.clientY;
      startRotation = state.rotation.slice();
      svgNode.classList.add("dragging");
      svgNode.setPointerCapture(event.pointerId);
    });

    svgNode.addEventListener("pointermove", (event) => {
      if (!state.dragging || !startRotation) return;
      const deltaX = event.clientX - startX;
      const deltaY = event.clientY - startY;
      state.rotation = [
        startRotation[0] + deltaX * DRAG_SENSITIVITY,
        clamp(startRotation[1] - deltaY * DRAG_SENSITIVITY, -82, 82),
        0,
      ];
      updatePaths();
    });

    function stopDrag(event) {
      if (!state.dragging) return;
      state.dragging = false;
      startRotation = null;
      svgNode.classList.remove("dragging");
      if (event.pointerId !== undefined && svgNode.hasPointerCapture(event.pointerId)) {
        svgNode.releasePointerCapture(event.pointerId);
      }
    }

    svgNode.addEventListener("pointerup", stopDrag);
    svgNode.addEventListener("pointercancel", stopDrag);
    svgNode.addEventListener("lostpointercapture", stopDrag);
  }

  function attachZoom(svgNode) {
    svgNode.addEventListener("wheel", (event) => {
      event.preventDefault();
      stopAnimation();
      const zoomFactor = Math.exp(-event.deltaY * ZOOM_SENSITIVITY);
      state.zoom = clamp(state.zoom * zoomFactor, MINIMUM_ZOOM, MAXIMUM_ZOOM);
      updatePaths();
    }, { passive: false });
  }

  function drawGlobe() {
    const container = document.getElementById("region-globe");
    if (!container || !state.options) return;
    if (!window.d3?.geoOrthographic || !window.topojson?.feature) {
      setStatus("Map unavailable");
      return;
    }

    try {
      const { activeRegion, regionMetadata } = state.options;
      const regionFillFeature = regionFillPolygon(activeRegion, regionMetadata);
      const regionOutlineFeature = regionOutline(activeRegion, regionMetadata);
      container.innerHTML = "";

      const projection = window.d3
        .geoOrthographic()
        .scale((HEIGHT - MARGIN * 2) / 2)
        .translate([WIDTH * GLOBE_CENTER_X_RATIO, HEIGHT / 2])
        .precision(0.6)
        .rotate(state.rotation || targetRotation(activeRegion, regionMetadata));
      const baseScale = projection.scale();
      const path = window.d3.geoPath(projection);
      const graticule = window.d3.geoGraticule10();

      const svg = appendSvgElement(container, "svg", {
        viewBox: `0 0 ${WIDTH} ${HEIGHT}`,
        role: "img",
        "aria-label": `${activeRegionLabel()} evaluation region on a rotatable globe`,
      });

      const defs = appendSvgElement(svg, "defs");
      const gradient = appendSvgElement(defs, "radialGradient", {
        id: "region-globe-ocean-gradient",
        cx: "34%",
        cy: "28%",
      });
      appendSvgElement(gradient, "stop", {
        class: "region-globe-ocean-stop-highlight",
        offset: "0%",
      });
      appendSvgElement(gradient, "stop", {
        class: "region-globe-ocean-stop",
        offset: "100%",
      });

      appendPath(svg, { type: "Sphere" }, "region-globe-ocean");
      appendPath(svg, graticule, "region-globe-graticule");

      if (regionFillFeature) {
        appendPath(svg, regionFillFeature, "region-globe-active-region");
      }

      if (state.landFeature) {
        appendPath(svg, state.landFeature, "region-globe-land");
      }

      if (regionOutlineFeature) {
        appendPath(svg, regionOutlineFeature, "region-globe-active-region-outline-halo");
        appendPath(svg, regionOutlineFeature, "region-globe-active-region-outline");
      }

      appendPath(svg, { type: "Sphere" }, "region-globe-outline");

      state.current = { baseScale, container, path, projection, svg };
      attachDrag(svg);
      attachZoom(svg);
      updatePaths();
    } catch (error) {
      console.warn("OceanBench region globe failed to render", error);
      state.current = null;
      setStatus("Map unavailable");
    }
  }

  function render(options) {
    if (!options?.activeRegion) return;

    const previousRegion = state.activeRegion;
    state.activeRegion = options.activeRegion;
    state.options = options;
    const nextRotation = targetRotation(options.activeRegion, options.regionMetadata);
    const nextZoom = targetZoom(options.activeRegion, options.regionMetadata);

    if (!state.rotation) {
      state.rotation = nextRotation;
      state.zoom = nextZoom;
    }

    drawGlobe();

    loadLandFeature().then((landFeature) => {
      if (!landFeature) return;
      drawGlobe();
      if (previousRegion !== options.activeRegion) {
        animateTo(nextRotation, nextZoom);
      }
    });

    if (previousRegion !== options.activeRegion) {
      animateTo(nextRotation, nextZoom);
    }
  }

  window.OceanBenchRegionGlobe = { render };
}());
