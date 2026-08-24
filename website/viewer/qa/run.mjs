#!/usr/bin/env node

// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

import { spawn } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const qaDir = path.dirname(fileURLToPath(import.meta.url));
const websiteDir = path.resolve(qaDir, "..", "..");
const viewerUrlPath = "/viewer/index.html?qa=1";

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
    name: "gpu-dpr2-1437",
    viewport: { width: 1437, height: 900 },
    deviceScaleFactor: 2,
  },
  {
    name: "nogpu-dpr1-1440",
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
    launchArgs: ["--disable-gpu"],
  },
  {
    name: "nogpu-dpr1-1437",
    viewport: { width: 1437, height: 900 },
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

async function runConfiguration(playwright, config, baseUrl) {
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

    await page.goto(`${baseUrl}${viewerUrlPath}`, { waitUntil: "load" });

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
    let warmScrubRequests = 0;
    const warmScrubUrls = [];
    page.on("request", (request) => {
      const url = request.url();
      if (url.startsWith("data:")) return;
      if (!/\/(c\/|chunks?\/|\d+\.\d+\.\d+)/.test(url) && !url.includes(".parquet")) return;
      warmScrubRequests += 1;
      warmScrubUrls.push(url);
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
    if (warmScrubRequests > 0) {
      failures.push(
        `warm back-scrub issued ${warmScrubRequests} field-data request(s), expected 0: ${warmScrubUrls
          .slice(0, 4)
          .map((url) => url.split("/").slice(-3).join("/"))
          .join(", ")}`,
      );
    }

    const expectations = JSON.parse(await readFile(path.join(qaDir, "expectations.json"), "utf8"));
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

    const filePath = path.join(qaDir, "expectations.json");
    const expectations = JSON.parse(await readFile(filePath, "utf8"));
    for (const probe of expectations.layoutProbes) {
      const box = await page.locator(probe.selector).boundingBox();
      if (!box || typeof box[probe.property] !== "number") {
        throw new Error(`seed probe ${probe.name}: element not visible (${probe.selector})`);
      }
      probe.expected = Number(box[probe.property].toFixed(2));
      console.log(`seed probe ${probe.name}: ${probe.expected}px`);
    }
    expectations.$note = [
      "Single editable place for QA expectations. Values are CSS pixels measured at the",
      "reference viewport (see viewport below).",
      "Re-baseline 2026-08-21: layout intentionally changed today (cover-fit map, legend chips);",
      "the previous 228px lead slider seed was stale. Reseed with: node qa/run.mjs --seed",
      "Tolerances are absolute pixel deltas; do not widen them to force green runs."
    ];
    await writeFile(filePath, `${JSON.stringify(expectations, null, 2)}\n`);
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

async function main() {
  const playwright = (await import("playwright")).default ?? (await import("playwright"));
  const exitCodes = [];

  if (process.argv.includes("--seed")) {
    const server = await startServer();
    try {
      process.stdout.write(`[seed] ${server.base}${viewerUrlPath}\n`);
      await seedExpectations(playwright, server.base);
      console.log("[seed] PASS: expectations.json rewritten");
    } catch (error) {
      console.error(`[seed] ERROR: ${error}`);
      const stderrText = server.getStderr();
      if (stderrText) {
        console.error(stderrText);
      }
      exitCodes.push(1);
    } finally {
      server.stop();
    }
    process.exit(exitCodes.some((code) => code !== 0) ? 1 : 0);
  }

  for (const config of MATRIX) {
    const server = await startServer();

    try {
      process.stdout.write(`[${config.name}] ${server.base}${viewerUrlPath}\n`);
      const failures = await runConfiguration(playwright, config, server.base);

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

  process.exit(exitCodes.some((code) => code !== 0) ? 1 : 0);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
