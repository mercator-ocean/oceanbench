// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Windy-style advected-particle current animation over a (uo, vo) velocity field
// (contracts.md §6: "animation only for motion — GPU current particles"). Runs on
// the 2D canvas with the classic fading-trail technique: every frame the whole
// overlay is dimmed by a translucent fill, then each live particle is advected by a
// bilinear sample of the vector field and drawn as a short glowing segment. Trails
// therefore fade over ~1 s, giving the streaming-flow look without storing history.
//
// The engine is projection- and data-agnostic: the panel hands it a small context
// object it reads afresh every frame (sampler, projection, viewport, theme, speed),
// so pan/zoom/lead-scrub/dataset-switch need no particle rebuild. Particle state is
// held in normalized world coordinates (x = (lon+180)/360, y = (90-lat)/180), so
// screen speed scales naturally with zoom — physically faithful — and 1/12° tiles
// drop in unchanged (finer sampler, identical advection).

import { sample as sampleColormap } from "../vendor/cmocean/colormaps.js";

// Advection gain: normalized-world units travelled per (m/s) per frame at speed 1,
// before the per-frame zoom scaling. Tuned low so flow reads: a strong ~0.5 m/s
// current takes several seconds to cross a visible eddy at 1× — comprehensible, not a
// blur. Multiplied by the user speed factor and by the visible world width each frame,
// so screen speed stays roughly constant as the user zooms (physically slower world
// step when zoomed in). Longer fade keeps trails on-screen so direction reads.
const ADVECTION_GAIN = 0.0016;
const BASE_PARTICLE_DENSITY = 1 / 950; // particles per css pixel² of visible panel
const MIN_PARTICLES = 500;
const MAX_PARTICLES = 6000;
const MAX_AGE_FRAMES = 170;
const TRAIL_FADE_ALPHA = 0.055;
const SPEED_COLORMAP = "speed";

function createParticle(random) {
  return { x: random(), y: random(), age: Math.floor(random() * MAX_AGE_FRAMES) };
}

/**
 * Start a particle animation writing into `canvas` (an overlay canvas sized in
 * device pixels). `context` is a live object the panel mutates; the engine reads:
 *   sampleVelocity(nx, ny) -> {u, v} | null   (null over land / outside data)
 *   project(nx, ny)        -> {x, y}          (device-pixel screen position)
 *   viewport               -> {minX, minY, maxX, maxY} normalized-world window shown
 *   magnitudeScale         -> m/s that maps to the top of the speed colormap
 *   theme, speed, devicePixelRatio, playing
 * Returns a handle with { stop(), resize(), reseed() }.
 */
export function startParticleField(canvas, context) {
  const drawing = canvas.getContext("2d");
  let particles = [];
  let animationHandle = null;
  let seededForArea = -1;

  const random = Math.random;

  function targetCount() {
    const view = context.viewport;
    const cssArea = canvas.width * canvas.height / (context.devicePixelRatio ** 2);
    const visibleFraction = Math.max(0.02, (view.maxX - view.minX) * (view.maxY - view.minY));
    // Denser when zoomed in (small visible fraction) so streamlines stay legible.
    const zoomBoost = 1 / Math.sqrt(visibleFraction);
    const count = Math.round(cssArea * BASE_PARTICLE_DENSITY * Math.min(3, zoomBoost));
    return Math.max(MIN_PARTICLES, Math.min(MAX_PARTICLES, count));
  }

  function reseed() {
    const count = targetCount();
    particles = new Array(count);
    for (let i = 0; i < count; i += 1) particles[i] = spawnInView();
    seededForArea = count;
  }

  function spawnInView() {
    const view = context.viewport;
    return {
      x: view.minX + random() * (view.maxX - view.minX),
      y: view.minY + random() * (view.maxY - view.minY),
      age: Math.floor(random() * MAX_AGE_FRAMES),
    };
  }

  function fadeTrails() {
    // Translucent fill dims previous frame — the trail memory. Composite mode
    // "destination-out" erases toward transparent so the field colour shows through.
    drawing.globalCompositeOperation = "destination-out";
    drawing.fillStyle = `rgba(0, 0, 0, ${TRAIL_FADE_ALPHA})`;
    drawing.fillRect(0, 0, canvas.width, canvas.height);
    drawing.globalCompositeOperation = "source-over";
  }

  function frame() {
    if (!context.playing) {
      animationHandle = requestAnimationFrame(frame);
      return;
    }
    const desired = targetCount();
    if (Math.abs(desired - seededForArea) > seededForArea * 0.25) reseed();

    fadeTrails();
    const view = context.viewport;
    // Scale the world step by the visible width so a particle crosses the viewport in
    // a similar wall-clock time at every zoom (screen speed stays comprehensible).
    const visibleWidth = Math.max(0.02, view.maxX - view.minX);
    const stepGain = ADVECTION_GAIN * context.speed * visibleWidth;
    const ratio = context.devicePixelRatio;
    const glow = context.theme === "light" ? 0.9 : 1;
    drawing.lineWidth = Math.max(1, 1.1 * ratio);
    drawing.lineCap = "round";

    for (const particle of particles) {
      particle.age += 1;
      const outside =
        particle.x < view.minX || particle.x > view.maxX || particle.y < view.minY || particle.y > view.maxY;
      if (particle.age > MAX_AGE_FRAMES || outside) {
        Object.assign(particle, spawnInView());
        continue;
      }
      const velocity = context.sampleVelocity(particle.x, particle.y);
      if (!velocity) {
        Object.assign(particle, spawnInView());
        continue;
      }
      const previousX = particle.x;
      const previousY = particle.y;
      // North-positive vo decreases normalized y (north is up).
      particle.x += velocity.u * stepGain;
      particle.y -= velocity.v * stepGain;

      const from = context.project(previousX, previousY);
      const to = context.project(particle.x, particle.y);
      const magnitude = Math.hypot(velocity.u, velocity.v);
      const normalized = Math.min(1, magnitude / context.magnitudeScale);
      if (context.muted) {
        // Over a muted/grayscale field the flow lines drop their speed colormap and
        // render theme-appropriate black-and-white (dark on light, light on dark), so
        // the overlay colours (obs points, eddy outlines, trajectory fans) stay the
        // only coloured marks. Speed still modulates brightness for legibility.
        const level = context.theme === "light" ? 90 - normalized * 60 : 165 + normalized * 90;
        const shade = Math.max(0, Math.min(255, Math.round(level)));
        drawing.strokeStyle = `rgba(${shade}, ${shade}, ${shade}, ${0.7 * glow})`;
      } else {
        const [r, g, b] = sampleColormap(SPEED_COLORMAP, 0.15 + normalized * 0.85);
        drawing.strokeStyle = `rgba(${r}, ${g}, ${b}, ${0.65 * glow})`;
      }
      drawing.beginPath();
      drawing.moveTo(from.x, from.y);
      drawing.lineTo(to.x, to.y);
      drawing.stroke();
    }
    animationHandle = requestAnimationFrame(frame);
  }

  function stop() {
    if (animationHandle !== null) cancelAnimationFrame(animationHandle);
    animationHandle = null;
    drawing.clearRect(0, 0, canvas.width, canvas.height);
  }

  function resize() {
    drawing.clearRect(0, 0, canvas.width, canvas.height);
    reseed();
  }

  reseed();
  animationHandle = requestAnimationFrame(frame);
  return { stop, resize, reseed };
}

/**
 * Bilinear velocity sampler over a decoded (u, v) field on a regular lat/lon grid.
 * Returns a function nx,ny (normalized world) -> {u, v} in m/s, or null over land.
 * Latitudes may be ascending or descending; the sampler registers on coordinates.
 */
export function makeVelocitySampler(uField, vField, latitudes, longitudes) {
  const width = uField.width;
  const height = uField.height;
  const lonMin = longitudes[0];
  const lonStep = longitudes.length > 1 ? longitudes[1] - longitudes[0] : 1;
  const latMin = latitudes[0];
  const latStep = latitudes.length > 1 ? latitudes[1] - latitudes[0] : 1;
  // A global field is periodic in longitude: sampling a wrapped copy (nx outside
  // [0,1], i.e. across the dateline) must fold back onto the data so particles keep
  // flowing on every visible copy. Regional fields (span well under 360°) do not wrap.
  const longitudeSpan = Math.abs(lonStep) * width;
  const periodic = longitudeSpan >= 359;

  return function sampleVelocity(nx, ny) {
    let lon = nx * 360 - 180;
    const lat = 90 - ny * 180;
    if (periodic) {
      lon = ((((lon - lonMin) % 360) + 360) % 360) + lonMin;
    }
    const fx = (lon - lonMin) / lonStep;
    const fy = (lat - latMin) / latStep;
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    if (x0 < 0 || x0 >= width - 1 || y0 < 0 || y0 >= height - 1) return null;
    const tx = fx - x0;
    const ty = fy - y0;
    const u = bilinear(uField.data, width, x0, y0, tx, ty);
    if (u === null) return null;
    const v = bilinear(vField.data, width, x0, y0, tx, ty);
    if (v === null) return null;
    return { u, v };
  };
}

function bilinear(data, width, x0, y0, tx, ty) {
  const a = data[y0 * width + x0];
  const b = data[y0 * width + x0 + 1];
  const c = data[(y0 + 1) * width + x0];
  const d = data[(y0 + 1) * width + x0 + 1];
  if (Number.isNaN(a) || Number.isNaN(b) || Number.isNaN(c) || Number.isNaN(d)) return null;
  const top = a + (b - a) * tx;
  const bottom = c + (d - c) * tx;
  return top + (bottom - top) * ty;
}

/** Speed-magnitude field (√(u²+v²)) for the currents-mode background, NaN over land. */
export function speedMagnitudeField(uField, vField) {
  const data = new Float32Array(uField.data.length);
  for (let i = 0; i < data.length; i += 1) {
    const u = uField.data[i];
    const v = vField.data[i];
    data[i] = Number.isNaN(u) || Number.isNaN(v) ? NaN : Math.hypot(u, v);
  }
  return { data, width: uField.width, height: uField.height };
}
