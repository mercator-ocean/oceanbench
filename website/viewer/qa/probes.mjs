// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Behavioural probes for the viewer QA harness.
//
// The layout probes in run.mjs answer "is the page alive and the right shape". The
// probes here answer "does the app draw the right thing", which is the question a
// refactor actually needs answered. They work by reading canvas pixels back out of the
// real page and hashing them, so a changed colormap, a dropped overlay, a stale frame
// or a diverged decode path all show up as a changed hash.
//
// INVARIANT: fingerprint baselines in fingerprints.json are frozen truth for a refactor.
// A refactor that changes a fingerprint is a refactor that changed what the user sees.
// Reseed only when the visual change is intentional, and say so in the run report.

const FINGERPRINT_SAMPLE_INTERVAL_MILLISECONDS = 400;
const FINGERPRINT_STABLE_SAMPLES = 4;
const FINGERPRINT_SETTLE_ATTEMPTS = 120;
const FINGERPRINT_READY_TIMEOUT_MILLISECONDS = 45000;
// A blank page is perfectly stable, so stability alone is not readiness: the loop below
// would happily certify the 300x150 default canvas the browser hands out before the first
// layout. Sampling starts only once the app has laid a panel out, drawn into it, and (for
// an overlay state) put something on the overlay canvas.
const FINGERPRINT_GRACE_MILLISECONDS = 1500;

// One panel geometry and one device pixel ratio for every fingerprint, so a hash
// difference means the drawing changed rather than the rasterizer.
export const FINGERPRINT_VIEWPORT = { width: 1440, height: 900 };
export const FINGERPRINT_DEVICE_SCALE_FACTOR = 1;

const BASE_HASH_PARAMETERS = {
  s: "0",
  z: "1.000",
  cx: "0.5000",
  cy: "0.5000",
  theme: "light",
  region: "global",
  rw: "352",
  cw: "256",
  play: "0",
  spd: "1.0",
};

function hashUrl(overrides) {
  const parameters = new URLSearchParams({ ...BASE_HASH_PARAMETERS, ...overrides });
  return `#${parameters.toString()}`;
}

const SINGLE_PANEL = "glonet,sea_surface_height_above_geoid,field";
const SECOND_PANEL = "glo12,sea_surface_height_above_geoid,field";

// Twelve states chosen to touch every drawing path the viewer has: the three panel
// display modes, the two map scopes, both year metrics, the two overlay renderers, a
// regional zoom, and the velocity path (with particles off, so the frame is static).
export const FINGERPRINT_STATES = [
  { name: "field-height-lead1", hash: hashUrl({ layout: "1", l: "1", p0: SINGLE_PANEL }) },
  { name: "field-height-lead10", hash: hashUrl({ layout: "1", l: "10", p0: SINGLE_PANEL }) },
  {
    name: "field-temperature-lead5",
    hash: hashUrl({ layout: "1", l: "5", p0: "glonet,sea_water_potential_temperature,field" }),
  },
  {
    name: "field-currents-lead3",
    hash: hashUrl({ layout: "1", l: "3", p0: "glonet,current_speed,field" }),
  },
  {
    name: "two-panel-side-lead1",
    hash: hashUrl({ layout: "2", dm: "side", l: "1", p0: SINGLE_PANEL, p1: SECOND_PANEL }),
  },
  {
    name: "two-panel-swipe-lead1",
    hash: hashUrl({ layout: "2", dm: "swipe", l: "1", p0: SINGLE_PANEL, p1: SECOND_PANEL }),
  },
  {
    name: "two-panel-difference-lead1",
    hash: hashUrl({ layout: "2", dm: "diff", l: "1", p0: SINGLE_PANEL, p1: SECOND_PANEL }),
  },
  {
    name: "overlay-class4-lead1",
    hash: hashUrl({ layout: "1", l: "1", ov: "class4", p0: SINGLE_PANEL }),
    requireOverlay: true,
  },
  {
    name: "overlay-eddies-lead1",
    hash: hashUrl({ layout: "1", l: "1", ov: "eddies", eref: "glorys", p0: SINGLE_PANEL }),
    requireOverlay: true,
  },
  { name: "year-scope-error", hash: hashUrl({ layout: "1", scope: "year", p0: SINGLE_PANEL }) },
  {
    name: "year-scope-bias",
    hash: hashUrl({ layout: "1", scope: "year", metric: "bias", p0: SINGLE_PANEL }),
  },
  {
    name: "region-ibi-zoomed",
    hash: hashUrl({ layout: "1", l: "1", region: "ibi", z: "6.000", cx: "0.4750", cy: "0.2600", p0: SINGLE_PANEL }),
  },
];

// FNV-1a over the raw RGBA bytes. Hashed inside the page because moving several
// megabytes of pixels per sample across the CDP bridge dominates the run otherwise.
function fingerprintScript() {
  const panels = Array.from(document.querySelectorAll(".panel"));
  const digest = (canvas) => {
    if (!canvas || !canvas.width || !canvas.height) return "empty";
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) return "no-context";
    const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
    let hash = 0x811c9dc5;
    for (let index = 0; index < data.length; index += 1) {
      hash ^= data[index];
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    return `${canvas.width}x${canvas.height}:${hash.toString(16).padStart(8, "0")}`;
  };
  return panels
    .map((panel, index) => {
      const field = digest(panel.querySelector(".panel-field"));
      const overlay = digest(panel.querySelector(".panel-overlay"));
      return `p${index}[field=${field},overlay=${overlay}]`;
    })
    .join(" ");
}

export async function readFingerprint(page) {
  return page.evaluate(fingerprintScript);
}

async function waitForRenderReady(page, { requireOverlay = false } = {}) {
  await page.waitForFunction(
    (needOverlay) => {
      const panel = document.querySelector(".panel");
      if (!panel) return false;
      const field = panel.querySelector(".panel-field");
      if (!field || field.width <= 300) return false;
      const loading = panel.querySelector(".panel-loading");
      if (loading && !loading.hidden) return false;
      if (!needOverlay) return true;
      const overlay = panel.querySelector(".panel-overlay");
      if (!overlay || overlay.width <= 300) return false;
      const { data } = overlay
        .getContext("2d", { willReadFrequently: true })
        .getImageData(0, 0, overlay.width, overlay.height);
      for (let index = 3; index < data.length; index += 4) {
        if (data[index] !== 0) return true;
      }
      return false;
    },
    requireOverlay,
    { timeout: FINGERPRINT_READY_TIMEOUT_MILLISECONDS, polling: 250 },
  );
  await page.waitForTimeout(FINGERPRINT_GRACE_MILLISECONDS);
}

// A render settles when the same fingerprint comes back several samples running. Polling
// for quiescence rather than waiting a fixed time keeps the probe honest on a slow
// network without making every fast state pay for the slowest one.
export async function waitForStableFingerprint(page, readyOptions) {
  await waitForRenderReady(page, readyOptions);
  let previous = null;
  let repeats = 0;
  for (let attempt = 0; attempt < FINGERPRINT_SETTLE_ATTEMPTS; attempt += 1) {
    const current = await readFingerprint(page);
    if (current === previous) {
      repeats += 1;
      if (repeats >= FINGERPRINT_STABLE_SAMPLES) return current;
    } else {
      previous = current;
      repeats = 1;
    }
    await page.waitForTimeout(FINGERPRINT_SAMPLE_INTERVAL_MILLISECONDS);
  }
  throw new Error(`fingerprint never settled (last ${previous})`);
}

export async function openFingerprintPage(context, baseUrl, viewerUrlPath, state) {
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error && error.stack ? error.stack : error)));
  await page.goto(`${baseUrl}${viewerUrlPath}${state.hash}`, { waitUntil: "load" });
  return { page, errors };
}

// P1: every state's drawn pixels against the frozen baseline.
export async function probeRenderFingerprints(context, baseUrl, viewerUrlPath, baseline, failures) {
  const measured = {};
  for (const state of FINGERPRINT_STATES) {
    const { page, errors } = await openFingerprintPage(context, baseUrl, viewerUrlPath, state);
    try {
      const fingerprint = await waitForStableFingerprint(page, { requireOverlay: state.requireOverlay });
      measured[state.name] = fingerprint;
      const expected = baseline ? baseline[state.name] : null;
      if (baseline && expected !== fingerprint) {
        failures.push(`render fingerprint ${state.name}: got ${fingerprint}, baseline ${expected}`);
      }
      for (const error of errors) failures.push(`render fingerprint ${state.name} page error: ${error}`);
    } catch (error) {
      failures.push(`render fingerprint ${state.name}: ${error.message}`);
    } finally {
      await page.close();
    }
  }
  return measured;
}

// P2: ten ArrowRight presses with no settling between them. pressAndSettle in run.mjs is
// specifically designed to avoid the render-token race; this fires into it on purpose and
// requires the app to land on the same pixels as a calm scrub to lead 10.
export async function probeFastScrubRace(context, baseUrl, viewerUrlPath, baseline, failures) {
  const target = baseline ? baseline["field-height-lead10"] : null;
  if (!target) return;
  const start = FINGERPRINT_STATES.find((state) => state.name === "field-height-lead1");
  const { page, errors } = await openFingerprintPage(context, baseUrl, viewerUrlPath, start);
  try {
    await waitForStableFingerprint(page);
    const slider = page.locator("#lead-day");
    await slider.focus();
    for (let press = 0; press < 9; press += 1) {
      await slider.press("ArrowRight", { delay: 0 });
    }
    const settled = await waitForStableFingerprint(page);
    const lead = Number(await slider.inputValue());
    if (lead !== 10) {
      failures.push(`fast-scrub race: slider landed on lead ${lead}, expected 10`);
    } else if (settled !== target) {
      failures.push(`fast-scrub race: settled on ${settled}, expected the lead-10 baseline ${target}`);
    }
    for (const error of errors) failures.push(`fast-scrub race page error: ${error}`);
  } catch (error) {
    failures.push(`fast-scrub race: ${error.message}`);
  } finally {
    await page.close();
  }
}

// P3: the colour range is grow-only within one selection signature (stable-ranges.js).
// Reading it back across a lead sweep is the only way to see the property that keeps the
// map from flickering as the slider moves.
export async function probeColorRangeMonotonicity(context, baseUrl, viewerUrlPath, failures) {
  const start = FINGERPRINT_STATES.find((state) => state.name === "field-height-lead1");
  const { page, errors } = await openFingerprintPage(context, baseUrl, viewerUrlPath, start);
  try {
    await waitForStableFingerprint(page);
    const readSpan = () =>
      page.evaluate(() => {
        const probe = window.oceanbenchViewerQaProbe;
        if (!probe) return null;
        const range = probe.colorRanges()[0];
        return range ? range[1] - range[0] : null;
      });
    const firstSpan = await readSpan();
    if (firstSpan === null) {
      failures.push("colour range: the ?qa= probe hook published no range for panel 1");
      return;
    }
    const signatureBefore = await page.evaluate(() => window.oceanbenchViewerQaProbe.selectionSignature());
    const slider = page.locator("#lead-day");
    await slider.focus();
    let widest = firstSpan;
    for (let lead = 2; lead <= 10; lead += 1) {
      await slider.press("ArrowRight");
      await waitForStableFingerprint(page);
      const span = await readSpan();
      if (span === null) {
        failures.push(`colour range: no range published at lead ${lead}`);
        return;
      }
      if (span < widest - 1e-9) {
        failures.push(`colour range shrank at lead ${lead}: ${span} < ${widest} within one selection`);
        return;
      }
      widest = span;
    }
    const signatureAfter = await page.evaluate(() => window.oceanbenchViewerQaProbe.selectionSignature());
    if (signatureAfter !== signatureBefore) {
      failures.push(`colour range: the selection signature moved during a lead sweep (${signatureBefore} -> ${signatureAfter})`);
    }
    for (const error of errors) failures.push(`colour range page error: ${error}`);
  } catch (error) {
    failures.push(`colour range: ${error.message}`);
  } finally {
    await page.close();
  }
}

// P5: the particle field advects while playing and holds perfectly still when paused. The
// premultiplied-alpha fade stall documented in particles.js is invisible to every other
// probe here, because it leaves the page error-free and the layout untouched.
export async function probeParticleLiveness(context, baseUrl, viewerUrlPath, failures) {
  const playing = {
    name: "particles-playing",
    hash: FINGERPRINT_STATES.find((state) => state.name === "field-currents-lead3").hash.replace("play=0", "play=1"),
  };
  const { page, errors } = await openFingerprintPage(context, baseUrl, viewerUrlPath, playing);
  try {
    const digestParticles = () =>
      page.evaluate(() => {
        const canvas = document.querySelector(".panel-particles");
        if (!canvas || !canvas.width) return "empty";
        const context2d = canvas.getContext("2d", { willReadFrequently: true });
        const { data } = context2d.getImageData(0, 0, canvas.width, canvas.height);
        let hash = 0x811c9dc5;
        for (let index = 0; index < data.length; index += 1) {
          hash ^= data[index];
          hash = Math.imul(hash, 0x01000193) >>> 0;
        }
        return hash.toString(16).padStart(8, "0");
      });
    await page.waitForTimeout(6000);
    const first = await digestParticles();
    await page.waitForTimeout(600);
    const second = await digestParticles();
    if (first === "empty" || first === second) {
      failures.push(`particle liveness: playing frames did not change (${first} then ${second})`);
    }
    await page.evaluate(() => document.querySelector(".bar-more")?.setAttribute("open", ""));
    await page.locator("#particles-play").uncheck();
    await page.waitForTimeout(1200);
    const paused = await digestParticles();
    await page.waitForTimeout(600);
    const pausedAgain = await digestParticles();
    if (paused !== pausedAgain) {
      failures.push(`particle liveness: paused frames kept changing (${paused} then ${pausedAgain})`);
    }
    for (const error of errors) failures.push(`particle liveness page error: ${error}`);
  } catch (error) {
    failures.push(`particle liveness: ${error.message}`);
  } finally {
    await page.close();
  }
}

// P6: the software DEFLATE decoder in zarr.js runs only on browsers without
// DecompressionStream, which means it is never exercised here unless we take the native
// one away. Same bytes in, same pixels out, or the two decode paths have diverged.
export async function probeSoftwareInflateEquivalence(browser, baseUrl, viewerUrlPath, baseline, failures) {
  const state = FINGERPRINT_STATES.find((candidate) => candidate.name === "field-height-lead1");
  const expected = baseline ? baseline[state.name] : null;
  if (!expected) return;
  const context = await browser.newContext({
    viewport: FINGERPRINT_VIEWPORT,
    deviceScaleFactor: FINGERPRINT_DEVICE_SCALE_FACTOR,
  });
  try {
    await context.addInitScript(() => {
      delete window.DecompressionStream;
    });
    const { page, errors } = await openFingerprintPage(context, baseUrl, viewerUrlPath, state);
    try {
      const fingerprint = await waitForStableFingerprint(page);
      if (fingerprint !== expected) {
        failures.push(`software inflate: ${fingerprint} differs from the native-decode baseline ${expected}`);
      }
      for (const error of errors) failures.push(`software inflate page error: ${error}`);
    } finally {
      await page.close();
    }
  } catch (error) {
    failures.push(`software inflate: ${error.message}`);
  } finally {
    await context.close();
  }
}

// P7: the house rule bans em dashes in shipped text, and no linter reads the rendered DOM.
export async function probeRenderedEmDashes(page, failures) {
  // The character itself is written as an escape so this file does not contain the
  // thing it forbids.
  const offenders = await page.evaluate(() => {
    const EM_DASH = String.fromCharCode(0x2014);
    const found = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const text = walker.currentNode.nodeValue || "";
      if (text.includes(EM_DASH)) found.push(text.trim().slice(0, 60));
    }
    return found;
  });
  for (const offender of offenders.slice(0, 3)) {
    failures.push(`em dash in rendered text: ${offender}`);
  }
}

// P4: classify what the page fetches instead of counting everything as one number. The
// old binary counter said "zero requests during a warm back-scrub", which was never the
// invariant anyone wanted: one-time lazy loads land in the same window and are fine,
// while a single refetched field chunk is not.
export const REQUEST_CLASSES = ["fieldChunk", "matchupParquet", "manifest", "insightJson", "other"];

export function classifyRequest(url) {
  if (url.startsWith("data:") || url.startsWith("blob:")) return null;
  if (url.includes(".parquet")) return "matchupParquet";
  if (/\.zmetadata|\.zarray|\.zattrs|viewer-manifest\.json/.test(url)) return "manifest";
  // Zarr v2 chunk keys have one index per dimension: three for older stores, four for the
  // (start, lead, latitude, longitude) pyramid levels and five for the water columns.
  if (/\/(c\/|chunks?\/)/.test(url) || /\/\d+(\.\d+){2,}(\?|$)/.test(url)) return "fieldChunk";
  if (url.endsWith(".json")) return "insightJson";
  return "other";
}

export function checkRequestBudget(counts, budget, label, failures) {
  for (const className of REQUEST_CLASSES) {
    const allowed = budget[className];
    if (typeof allowed !== "number") continue;
    if (counts[className] > allowed) {
      failures.push(`${label}: ${counts[className]} ${className} request(s), budget ${allowed}`);
    }
  }
}

// P7: an open More menu sits above the rail's resize and collapse handles, and a click
// inside it never reaches them. Clicking where each handle is must leave the rail as it was.
export async function probeMenuAboveHandles(context, baseUrl, viewerUrlPath, failures) {
  const page = await context.newPage();
  try {
    await page.goto(`${baseUrl}${viewerUrlPath}`);
    await page.waitForSelector(".bar-more > summary");
    await page.click(".bar-more > summary");
    const railState = () =>
      page.evaluate(() => {
        const rail = document.querySelector("#context-rail");
        return `${rail.classList.contains("collapsed")}:${Math.round(rail.getBoundingClientRect().width)}`;
      });
    const before = await railState();
    const menu = await page.locator(".bar-more-menu").boundingBox();
    let overlapping = 0;
    for (const selector of ["#context-rail > .drawer-tab", "#context-rail > .drawer-resizer"]) {
      const handle = await page.locator(selector).boundingBox();
      if (!handle || !menu) continue;
      const x = handle.x + handle.width / 2;
      const top = Math.max(handle.y, menu.y + 4);
      const bottom = Math.min(handle.y + handle.height, menu.y + menu.height - 4);
      const y = (top + bottom) / 2;
      if (bottom <= top || x < menu.x || x > menu.x + menu.width) continue;
      overlapping += 1;
      const topmost = await page.evaluate(([px, py]) => {
        const element = document.elementFromPoint(px, py);
        return Boolean(element?.closest(".bar-more-menu"));
      }, [x, y]);
      if (!topmost) failures.push(`menu stacking: ${selector} is above the open More menu`);
      await page.mouse.move(x, y);
      await page.mouse.down();
      await page.mouse.move(x - 60, y);
      await page.mouse.up();
      if (!(await page.locator(".bar-more").getAttribute("open") === "")) await page.click(".bar-more > summary");
    }
    if (overlapping === 0) failures.push("menu stacking: no rail handle lies under the open More menu, probe is vacuous");
    const after = await railState();
    if (after !== before) failures.push(`menu stacking: rail changed under the open More menu (${before} -> ${after})`);
  } catch (error) {
    failures.push(`menu stacking: ${error.message}`);
  } finally {
    await page.close();
  }
}

// P8: every overlay opened from More keeps the More menu, the bar note and the bar groups
// fully on screen, at desktop, laptop and phone widths. Choosing an overlay used to let
// the note take the whole bar row, which threw More (and its menu) off the left edge.
// The note must also leave the maps' name strips uncovered.
export async function probeOverlayPanelsInViewport(browser, baseUrl, viewerUrlPath, failures) {
  const sizes = [
    { width: 1440, height: 900 },
    { width: 1280, height: 720 },
    { width: 390, height: 844 },
  ];
  for (const viewport of sizes) {
    const page = await browser.newPage({ viewport });
    try {
      await page.goto(`${baseUrl}${viewerUrlPath}`);
      await page.waitForSelector(".bar-more > summary", { state: "attached" });
      if (viewport.width < 600) await page.click("#filters-toggle");
      await page.click(".bar-more > summary");
      const modes = await page.$$eval("#overlay-mode option", (options) =>
        options.filter((option) => !option.disabled && !option.hidden && option.value !== "none").map((option) => option.value),
      );
      for (const mode of modes) {
        await page.selectOption("#overlay-mode", mode);
        await page.waitForTimeout(1500);
        const outside = await page.evaluate(() => {
          const width = document.documentElement.clientWidth;
          const height = window.innerHeight;
          const boxes = [".bar-more-menu", "#overlay-note", ".bar-more > summary"];
          return boxes.flatMap((selector) => {
            const element = document.querySelector(selector);
            if (!element || element.offsetParent === null || !element.textContent.trim()) return [];
            const box = element.getBoundingClientRect();
            const inside = box.left >= -0.5 && box.top >= -0.5 && box.right <= width + 0.5 && box.bottom <= height + 0.5;
            return inside ? [] : [`${selector} at ${Math.round(box.left)},${Math.round(box.top)}-${Math.round(box.right)},${Math.round(box.bottom)}`];
          });
        });
        const covered = await page.evaluate(() => {
          const note = document.querySelector("#overlay-note");
          if (!note || note.offsetParent === null || !note.textContent.trim()) return [];
          const n = note.getBoundingClientRect();
          return [...document.querySelectorAll(".panel-chips")]
            .filter((strip) => strip.offsetParent !== null)
            .map((strip) => strip.getBoundingClientRect())
            .filter((b) => b.width > 0 && b.left < n.right && n.left < b.right && b.top < n.bottom - 0.5 && n.top < b.bottom - 0.5)
            .map((b) => `name strip at ${Math.round(b.left)},${Math.round(b.top)} under the note`);
        });
        for (const entry of covered) failures.push(`overlay note covers a map (${viewport.width}x${viewport.height}, ${mode}): ${entry}`);
        for (const entry of outside) failures.push(`overlay panel off screen (${viewport.width}x${viewport.height}, ${mode}): ${entry}`);
      }
    } catch (error) {
      failures.push(`overlay panel in viewport (${viewport.width}): ${error.message}`);
    } finally {
      await page.close();
    }
  }
}
