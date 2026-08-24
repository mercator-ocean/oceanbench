// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Windy-style advected-particle current animation over a (uo, vo) velocity field
// (contracts.md §6: "animation only for motion, GPU current particles"). Runs on
// the 2D canvas with the classic fading-trail technique: every frame the whole
// overlay is dimmed by a translucent fill, then each live particle is advected by a
// bilinear sample of the vector field and drawn as a short glowing segment. Trails
// therefore fade over ~1 s, giving the streaming-flow look without storing history.
//
// The engine is projection- and data-agnostic: the panel hands it a small context
// object it reads afresh every frame (sampler, projection, viewport, theme, speed),
// so pan/zoom/lead-scrub/dataset-switch need no particle rebuild. Particle state is
// held in normalized world coordinates (x = (lon+180)/360, y = (90-lat)/180), so
// screen speed scales naturally with zoom, physically faithful, and 1/12° tiles
// drop in unchanged (finer sampler, identical advection).

import { sample as sampleColormap } from "../vendor/cmocean/colormaps.js";

// Advection gain: normalized-world units travelled per (m/s) per frame at speed 1,
// before the per-frame zoom scaling. Tuned low so flow reads: a strong ~0.5 m/s
// current takes several seconds to cross a visible eddy at 1×, comprehensible, not a
// blur. Multiplied by the user speed factor and by the visible world width each frame,
// so screen speed stays roughly constant as the user zooms (physically slower world
// step when zoomed in). Longer fade keeps trails on-screen so direction reads.
const ADVECTION_GAIN = 0.0016;
const BASE_PARTICLE_DENSITY = 1 / 950; // particles per css pixel² of visible panel
const MIN_PARTICLES = 500;
const MAX_PARTICLES = 6000;
const MAX_AGE_FRAMES = 170;
const TRAIL_FADE_ALPHA = 0.055;
// A trail is drawn at alpha 0.72 to 0.80 and dimmed by TRAIL_FADE_ALPHA every frame, so
// it is below one 8-bit level after about 102 frames. Nothing older than that contributes
// anything a viewer can see.
const TRAIL_VISIBLE_FRAMES = 102;
// The canvas keeps premultiplied 8-bit alpha and "destination-out" rounds a*(1-alpha)
// back up to a once a*alpha drops below 0.5, so the fade stalls: at TRAIL_FADE_ALPHA the
// alpha channel sticks at 9/255 and never reaches zero (measured identically on the
// software rasterizer and on ANGLE Metal). Stalled pixels unpremultiply to white, so
// every pixel a trail ever crossed kept a permanent pale veil, which is what made land
// read as two greys. Multiplying can never reach zero, so two buffers take turns
// instead: both receive every segment, one is on screen while the other warms up, and
// every TRAIL_RESET_FRAMES they swap and the one leaving the screen is hard-cleared.
// The period is comfortably longer than a trail's visible life, so the buffer coming on
// screen already holds a complete set of trails and the swap cannot be seen.
const TRAIL_RESET_FRAMES = Math.ceil(TRAIL_VISIBLE_FRAMES * 1.4);
const SPEED_COLORMAP = "speed";
// Streak ink contrast. The background field is colourized with the same speed ramp over
// [0, magnitudeScale], so tinting a streak straight from its own speed paints it onto its
// own colour and it vanishes (violet on violet through most of the ramp). Instead the ink
// reads the exact background colour under the segment and mixes hard toward white over the
// pale slow end and toward black over the deep fast blues, keeping a trace of hue while
// guaranteeing luminance separation from the field underneath without covering it.
const STREAK_INK_MIX = 0.8;
const STREAK_LUMINANCE_SPLIT = 128;
const STREAK_ALPHA = 0.8;

function streakInk(background) {
  const [r, g, b] = background;
  const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const target = luminance >= STREAK_LUMINANCE_SPLIT ? 0 : 255;
  const mix = (channel) => Math.round(channel + (target - channel) * STREAK_INK_MIX);
  return `${mix(r)}, ${mix(g)}, ${mix(b)}`;
}

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
  let stopped = false;
  // Trail accumulation happens off screen in two buffers; the visible canvas only ever
  // receives a blit of whichever one is currently on duty.
  let trailBuffers = [];
  let displayedBuffer = 0;
  let framesSinceSwap = 0;
  let projectionKey = "";
  // Segments of the current frame, replayed into both buffers so their histories match.
  const segmentPoints = new Float32Array(MAX_PARTICLES * 4);
  const segmentStyles = new Array(MAX_PARTICLES);

  const random = Math.random;

  function makeTrailBuffers() {
    trailBuffers = [0, 1].map(() => {
      const buffer = document.createElement("canvas");
      buffer.width = canvas.width;
      buffer.height = canvas.height;
      return buffer.getContext("2d");
    });
    displayedBuffer = 0;
    framesSinceSwap = 0;
  }

  function clearTrails() {
    for (const buffer of trailBuffers) buffer.clearRect(0, 0, canvas.width, canvas.height);
    drawing.clearRect(0, 0, canvas.width, canvas.height);
    framesSinceSwap = 0;
  }

  // The projection is the panel's, so read it rather than the viewport: the viewport is
  // clamped to the poles and does not move on every pan, while two projected corners
  // pin down origin and scale exactly. A pan or a zoom therefore drops the trails on the
  // frame it happens, instead of dragging a screen-space smear across the new land.
  function currentProjectionKey() {
    const topLeft = context.project(0, 0);
    const bottomRight = context.project(1, 1);
    return `${topLeft.x},${topLeft.y},${bottomRight.x},${bottomRight.y},${canvas.width},${canvas.height}`;
  }

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

  function fadeTrails(buffer) {
    // Translucent fill dims previous frame, the trail memory. Composite mode
    // "destination-out" erases toward transparent so the field colour shows through.
    buffer.globalCompositeOperation = "destination-out";
    buffer.fillStyle = `rgba(0, 0, 0, ${TRAIL_FADE_ALPHA})`;
    buffer.fillRect(0, 0, canvas.width, canvas.height);
    buffer.globalCompositeOperation = "source-over";
  }

  function paintSegments(buffer, count, lineWidth) {
    buffer.lineWidth = lineWidth;
    buffer.lineCap = "round";
    for (let i = 0; i < count; i += 1) {
      buffer.strokeStyle = segmentStyles[i];
      buffer.beginPath();
      buffer.moveTo(segmentPoints[i * 4], segmentPoints[i * 4 + 1]);
      buffer.lineTo(segmentPoints[i * 4 + 2], segmentPoints[i * 4 + 3]);
      buffer.stroke();
    }
  }

  function frame() {
    // Park the loop instead of burning a frame on nothing: a paused layer or a
    // backgrounded tab lets go of the rAF entirely, and resume() picks it back up.
    if (!context.playing || document.hidden) {
      animationHandle = null;
      return;
    }
    const desired = targetCount();
    if (Math.abs(desired - seededForArea) > seededForArea * 0.25) reseed();

    if (trailBuffers.length === 0 || trailBuffers[0].canvas.width !== canvas.width
      || trailBuffers[0].canvas.height !== canvas.height) {
      makeTrailBuffers();
    }
    const key = currentProjectionKey();
    if (key !== projectionKey) {
      projectionKey = key;
      clearTrails();
    }
    for (const buffer of trailBuffers) fadeTrails(buffer);
    const view = context.viewport;
    // Scale the world step by the visible width so a particle crosses the viewport in
    // a similar wall-clock time at every zoom (screen speed stays comprehensible).
    const visibleWidth = Math.max(0.02, view.maxX - view.minX);
    const stepGain = ADVECTION_GAIN * context.speed * visibleWidth;
    const ratio = context.devicePixelRatio;
    const glow = context.theme === "light" ? 0.9 : 1;
    const lineWidth = Math.max(1, 1.1 * ratio);
    let segmentCount = 0;

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
      // Where the step lands decides whether it may be drawn at all. Checking only the
      // origin let the last segment of a life reach into a land cell and leave ink on
      // the coast, so a particle whose next position has no velocity respawns silently.
      if (!context.sampleVelocity(particle.x, particle.y)) {
        Object.assign(particle, spawnInView());
        continue;
      }

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
        segmentStyles[segmentCount] = `rgba(${shade}, ${shade}, ${shade}, ${0.7 * glow})`;
      } else {
        // The field pixel under this segment is this same ramp position: the background
        // range tops out at magnitudeScale, so normalized is the background coordinate.
        const ink = streakInk(sampleColormap(SPEED_COLORMAP, normalized));
        segmentStyles[segmentCount] = `rgba(${ink}, ${STREAK_ALPHA * glow})`;
      }
      segmentPoints[segmentCount * 4] = from.x;
      segmentPoints[segmentCount * 4 + 1] = from.y;
      segmentPoints[segmentCount * 4 + 2] = to.x;
      segmentPoints[segmentCount * 4 + 3] = to.y;
      segmentCount += 1;
    }

    for (const buffer of trailBuffers) paintSegments(buffer, segmentCount, lineWidth);

    framesSinceSwap += 1;
    if (framesSinceSwap >= TRAIL_RESET_FRAMES) {
      const retiring = displayedBuffer;
      displayedBuffer = 1 - displayedBuffer;
      trailBuffers[retiring].clearRect(0, 0, canvas.width, canvas.height);
      framesSinceSwap = 0;
    }

    drawing.clearRect(0, 0, canvas.width, canvas.height);
    drawing.drawImage(trailBuffers[displayedBuffer].canvas, 0, 0);
    // The panel knows where it drew land; let it erase anything the flow put there.
    if (context.punchLand) context.punchLand(drawing);
    animationHandle = requestAnimationFrame(frame);
  }

  function resume() {
    if (stopped || animationHandle !== null) return;
    animationHandle = requestAnimationFrame(frame);
  }

  function onVisibilityChange() {
    if (!document.hidden) resume();
  }

  function stop() {
    stopped = true;
    document.removeEventListener("visibilitychange", onVisibilityChange);
    if (animationHandle !== null) cancelAnimationFrame(animationHandle);
    animationHandle = null;
    drawing.clearRect(0, 0, canvas.width, canvas.height);
    trailBuffers = [];
  }

  function resize() {
    makeTrailBuffers();
    drawing.clearRect(0, 0, canvas.width, canvas.height);
    reseed();
    resume();
  }

  reseed();
  document.addEventListener("visibilitychange", onVisibilityChange);
  animationHandle = requestAnimationFrame(frame);
  return { stop, resize, reseed, resume };
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
