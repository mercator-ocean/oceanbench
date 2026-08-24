#!/usr/bin/env node

// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

import { spawn } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  FINGERPRINT_DEVICE_SCALE_FACTOR,
  FINGERPRINT_VIEWPORT,
  REQUEST_CLASSES,
  checkRequestBudget,
  classifyRequest,
  probeColorRangeMonotonicity,
  probeFastScrubRace,
  probeParticleLiveness,
  probeRenderFingerprints,
  probeRenderedEmDashes,
  probeSoftwareInflateEquivalence,
} from "./probes.mjs";

const qaDir = path.dirname(fileURLToPath(import.meta.url));
const websiteDir = path.resolve(qaDir, "..", "..");
const viewerUrlPath = "/viewer/index.html?qa=1";
const fingerprintPath = path.join(qaDir, "fingerprints.json");
const expectationsPath = path.join(qaDir, "expectations.json");

// Six rendering-configuration cells and two behavioural cells. The matrix used to spend
// all eight on device pixel ratio crossed with viewport width, which bought four ways to
// discover the same rounding drift; trading two of those for a swipe cell and an overlay
// cell covers code paths nothing else in the harness ever entered, at the same runtime.
const MATRIX = [
  {
    name: "gpu-dpr1-1440",
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
  },
  {
    name: "gpu-dpr1-1437",
    viewport: { width: 1437, height: 900 },
    deviceScaleFactor: 1,
  },
  {
    name: "gpu-dpr2-1440",
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
  },
  {
    name: "nogpu-dpr1-1440",
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
    launchArgs: ["--disable-gpu"],
  },
  {
    name: "nogpu-dpr2-1440",
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
    launchArgs: ["--disable-gpu"],
  },
  {
    name: "nogpu-dpr2-1437",
    viewport: { width: 1437, height: 900 },
    deviceScaleFactor: 2,
    launchArgs: ["--disable-gpu"],
  },
  {
    name: "behaviour-swipe",
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
    hash: "#layout=2&dm=swipe&s=0&l=1&theme=light&region=global&play=0",
    // Two panels compose into one canvas here, so the single-panel width expectation
    // does not describe this cell. The fingerprint states cover its geometry instead.
    skipLayoutProbes: true,
  },
  {
    name: "behaviour-overlay",
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
    hash: "#layout=1&ov=class4&s=0&l=1&theme=light&region=global&play=0",
  },
];

function getFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

function isToleratedConsoleError(message, locationUrl) {
  return (
    message.includes("404") &&
    typeof locationUrl === "string" &&
    locationUrl.includes("viewer-config.json")
  );
}

function emptyRequestCounts() {
  return Object.fromEntries(REQUEST_CLASSES.map((className) => [className, 0]));
}

async function runConfiguration(playwright, config, baseUrl, expectations) {
  let browser;
  const failures = [];

  try {
    browser = await playwright.chromium.launch({
      headless: true,
      ...(config.launchArgs ? { args: config.launchArgs } : {}),
    });
    const context = await browser.newContext({
      viewport: config.viewport,
      deviceScaleFactor: config.deviceScaleFactor,
    });
    const page = await context.newPage();

    const pageErrors = [];
    const consoleErrors = [];

    page.on("pageerror", (error) => {
      pageErrors.push(String(error && error.stack ? error.stack : error));
    });

    page.on("console", (message) => {
      if (message.type() !== "error") {
        return;
      }
      const location = message.location() || {};
      const entry = {
        text: message.text(),
        url: location.url || "",
      };
      if (!isToleratedConsoleError(entry.text, entry.url)) {
        consoleErrors.push(entry);
      }
    });

    await page.goto(`${baseUrl}${viewerUrlPath}${config.hash || ""}`, { waitUntil: "load" });

    const slider = page.locator("#lead-day");
    await scrubToLeadTen(page, slider);
    const finalLead = Number(await slider.inputValue());
    if (finalLead !== 10) {
      failures.push(`lead scrub ended at ${finalLead}, expected 10`);
    }

    // Let the forward scrub's tail (prefetch of neighbouring leads, lazy chart loads)
    // finish before arming the counter, and count only field-chunk reads: the invariant
    // is "a warm back-scrub re-reads no field data", not "the app goes silent".
    await page.waitForTimeout(4000);
    const warmScrubCounts = emptyRequestCounts();
    const warmScrubUrls = [];
    page.on("request", (request) => {
      const className = classifyRequest(request.url());
      if (!className) return;
      warmScrubCounts[className] += 1;
      warmScrubUrls.push(`${className}:${request.url().split("/").slice(-3).join("/")}`);
    });

    let backValue = Number(await slider.inputValue());
    while (backValue > 1) {
      backValue = await pressAndSettle(page, slider, "ArrowLeft");
    }
    await page.waitForTimeout(250);
    const backLead = Number(await slider.inputValue());
    if (backLead !== 1) {
      failures.push(`back scrub ended at ${backLead}, expected 1`);
    }
    const budgetFailures = [];
    checkRequestBudget(warmScrubCounts, expectations.warmBackScrubBudget, "warm back-scrub", budgetFailures);
    for (const failure of budgetFailures) {
      failures.push(`${failure} (${warmScrubUrls.slice(0, 4).join(", ")})`);
    }
    console.log(
      `[${config.name}] warm back-scrub requests: ${REQUEST_CLASSES.map(
        (className) => `${className}=${warmScrubCounts[className]}`,
      ).join(" ")}`,
    );

    if (!config.skipLayoutProbes) {
      for (const probe of expectations.layoutProbes) {
        const box = await page.locator(probe.selector).boundingBox();
        if (!box) {
          failures.push(`layout probe ${probe.name}: element not visible`);
          continue;
        }
        const measured = box[probe.property];
        const delta = Math.abs(measured - probe.expected);
        if (delta > probe.tolerancePx) {
          failures.push(
            `layout probe ${probe.name}: ${probe.property} ${measured.toFixed(2)}px, expected ${probe.expected}px +/-${probe.tolerancePx}`,
          );
        } else {
          console.log(`[${config.name}] probe ${probe.name}: ${measured.toFixed(2)}px ok`);
        }
      }
    }

    await probeRenderedEmDashes(page, failures);

    if (pageErrors.length > 0) {
      for (const error of pageErrors) {
        failures.push(`page error: ${error}`);
      }
    }
    for (const error of consoleErrors) {
      failures.push(`console error (${error.url}): ${error.text}`);
    }
  } finally {
    if (browser) {
      await browser.close();
    }
  }

  return failures;
}

async function startServer() {
  const port = await getFreePort();
  const server = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], {
    cwd: websiteDir,
    stdio: ["ignore", "ignore", "pipe"],
  });
  let serverStderr = "";
  server.stderr.on("data", (chunk) => {
    serverStderr += chunk;
  });
  await new Promise((resolve, reject) => {
    server.once("spawn", resolve);
    server.once("error", reject);
  });
  return {
    process: server,
    base: `http://127.0.0.1:${port}`,
    getStderr: () => serverStderr,
    stop() {
      if (server.exitCode === null && !server.killed) {
        server.kill("SIGTERM");
      }
    },
  };
}

async function pressAndSettle(page, slider, key) {
  const before = Number(await slider.inputValue());
  await slider.press(key);
  for (let attempt = 0; attempt < 80; attempt += 1) {
    const current = Number(await slider.inputValue());
    if (current !== before) {
      return current;
    }
    await page.waitForTimeout(25);
  }
  throw new Error(`slider did not respond to ${key} press (stuck at ${before})`);
}

async function scrubToLeadTen(page, slider) {
  await slider.focus();
  let value = Number(await slider.inputValue());
  while (value < 10) {
    value = await pressAndSettle(page, slider, "ArrowRight");
  }
}

async function readFingerprintBaseline() {
  try {
    return JSON.parse(await readFile(fingerprintPath, "utf8")).states;
  } catch (error) {
    return null;
  }
}

// One browser, one context, one geometry for the whole behavioural pass: the states share
// an HTTP cache, and a fingerprint that moved because the device pixel ratio moved would
// tell nobody anything.
async function runBehaviouralPass(playwright, baseUrl, { seed }) {
  const failures = [];
  const baseline = seed ? null : await readFingerprintBaseline();
  if (!seed && !baseline) {
    failures.push("no fingerprint baseline: run `node qa/run.mjs --seed-fingerprints` first");
    return { failures, measured: null };
  }

  const browser = await playwright.chromium.launch({ headless: true });
  let measured = null;
  try {
    const context = await browser.newContext({
      viewport: FINGERPRINT_VIEWPORT,
      deviceScaleFactor: FINGERPRINT_DEVICE_SCALE_FACTOR,
    });
    measured = await probeRenderFingerprints(context, baseUrl, viewerUrlPath, baseline, failures);
    if (!seed) {
      await probeFastScrubRace(context, baseUrl, viewerUrlPath, baseline, failures);
      await probeColorRangeMonotonicity(context, baseUrl, viewerUrlPath, failures);
      await probeParticleLiveness(context, baseUrl, viewerUrlPath, failures);
      await context.close();
      await probeSoftwareInflateEquivalence(browser, baseUrl, viewerUrlPath, baseline, failures);
    } else {
      await context.close();
    }
  } finally {
    await browser.close();
  }
  return { failures, measured };
}

async function seedFingerprints(playwright, baseUrl) {
  const { failures, measured } = await runBehaviouralPass(playwright, baseUrl, { seed: true });
  if (failures.length > 0) {
    for (const failure of failures) console.error(`[seed-fingerprints] ${failure}`);
    return false;
  }
  await writeFile(
    fingerprintPath,
    `${JSON.stringify(
      {
        $note: [
          "Frozen render fingerprints: FNV-1a over the RGBA bytes of every panel's field and",
          "overlay canvas, per viewer state, measured at 1440x900 dpr 1 in headless Chromium.",
          "A refactor must leave every value here byte-identical. Reseed with",
          "`node qa/run.mjs --seed-fingerprints` ONLY when the drawing intentionally changed,",
          "and say which states moved and why in the change description.",
          "The values depend on the published viewer data at the configured data root, so they",
          "also move when that data is republished.",
        ],
        viewport: FINGERPRINT_VIEWPORT,
        deviceScaleFactor: FINGERPRINT_DEVICE_SCALE_FACTOR,
        states: measured,
      },
      null,
      2,
    )}\n`,
  );
  for (const [name, value] of Object.entries(measured)) {
    console.log(`seed fingerprint ${name}: ${value}`);
  }
  return true;
}

async function seedExpectations(playwright, baseUrl) {
  let browser;
  try {
    browser = await playwright.chromium.launch({ headless: true });
    const context = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      deviceScaleFactor: 1,
    });
    const page = await context.newPage();
    await page.goto(`${baseUrl}${viewerUrlPath}`, { waitUntil: "load" });

    const slider = page.locator("#lead-day");
    await scrubToLeadTen(page, slider);

    const expectations = JSON.parse(await readFile(expectationsPath, "utf8"));
    for (const probe of expectations.layoutProbes) {
      const box = await page.locator(probe.selector).boundingBox();
      if (!box || typeof box[probe.property] !== "number") {
        throw new Error(`seed probe ${probe.name}: element not visible (${probe.selector})`);
      }
      probe.expected = Number(box[probe.property].toFixed(2));
      console.log(`seed probe ${probe.name}: ${probe.expected}px`);
    }
    await writeFile(expectationsPath, `${JSON.stringify(expectations, null, 2)}\n`);
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

async function withServer(label, run) {
  const server = await startServer();
  try {
    process.stdout.write(`[${label}] ${server.base}${viewerUrlPath}\n`);
    return await run(server.base);
  } catch (error) {
    console.error(`[${label}] ERROR: ${error}`);
    const stderrText = server.getStderr();
    if (stderrText) {
      console.error(stderrText);
    }
    return null;
  } finally {
    server.stop();
  }
}

async function main() {
  const playwright = (await import("playwright")).default ?? (await import("playwright"));
  const exitCodes = [];

  if (process.argv.includes("--seed")) {
    const outcome = await withServer("seed", async (base) => {
      await seedExpectations(playwright, base);
      console.log("[seed] PASS: expectations.json rewritten");
      return true;
    });
    process.exit(outcome ? 0 : 1);
  }

  if (process.argv.includes("--seed-fingerprints")) {
    const outcome = await withServer("seed-fingerprints", (base) => seedFingerprints(playwright, base));
    if (outcome) console.log("[seed-fingerprints] PASS: fingerprints.json rewritten");
    process.exit(outcome ? 0 : 1);
  }

  const expectations = JSON.parse(await readFile(expectationsPath, "utf8"));

  for (const config of MATRIX) {
    const server = await startServer();

    try {
      process.stdout.write(`[${config.name}] ${server.base}${viewerUrlPath}${config.hash || ""}\n`);
      const failures = await runConfiguration(playwright, config, server.base, expectations);

      if (failures.length === 0) {
        console.log(`[${config.name}] PASS`);
        exitCodes.push(0);
      } else {
        for (const failure of failures) {
          console.error(`[${config.name}] FAIL: ${failure}`);
        }
        exitCodes.push(1);
      }
    } catch (error) {
      console.error(`[${config.name}] ERROR: ${error}`);
      const stderrText = server.getStderr();
      if (stderrText) {
        console.error(stderrText);
      }
      exitCodes.push(1);
    } finally {
      server.stop();
    }
  }

  const behaviouralOutcome = await withServer("behaviour", (base) =>
    runBehaviouralPass(playwright, base, { seed: false }),
  );
  if (!behaviouralOutcome) {
    exitCodes.push(1);
  } else if (behaviouralOutcome.failures.length === 0) {
    console.log("[behaviour] PASS");
    exitCodes.push(0);
  } else {
    for (const failure of behaviouralOutcome.failures) {
      console.error(`[behaviour] FAIL: ${failure}`);
    }
    exitCodes.push(1);
  }

  process.exit(exitCodes.some((code) => code !== 0) ? 1 : 0);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
