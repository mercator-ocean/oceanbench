# Viewer QA Harness

Permanent regression harness for the static viewer (`website/viewer/`). It boots a local static server, drives the real page in headless Chromium under Playwright, and asserts a fixed set of behavioral and layout invariants. It exists so that layout and interaction regressions are caught mechanically instead of by eye.

## Running

```sh
node qa/run.mjs
```

Runs all 8 matrix cells sequentially and exits non-zero if any cell reports failures.

```sh
node qa/run.mjs --seed
```

Re-baselines `expectations.json`: it loads the page once at the reference viewport (1440x900, dpr 1), measures each layout probe, and rewrites the `expected` values. Use this ONLY after an INTENTIONAL layout change. Do not widen tolerances to force green runs, and do not reseed to make an accidental regression pass.

Requires `playwright` (with Chromium installed) and `python3` for the static file server. The server binds to a random free port on 127.0.0.1 and serves the `website/` directory; the page under test is `/viewer/index.html?qa=1`.

## Matrix

Eight cells covering GPU on/off x device pixel ratio 1/2 x viewport width 1440/1437 (height always 900):

| Cell              | GPU      | DPR | Width |
| ----------------- | -------- | --- | ----- |
| gpu-dpr1-1440     | enabled  | 1   | 1440  |
| gpu-dpr1-1437     | enabled  | 1   | 1437  |
| gpu-dpr2-1440     | enabled  | 2   | 1440  |
| gpu-dpr2-1437     | enabled  | 2   | 1437  |
| nogpu-dpr1-1440   | disabled | 1   | 1440  |
| nogpu-dpr1-1437   | disabled | 1   | 1437  |
| nogpu-dpr2-1440   | disabled | 2   | 1440  |
| nogpu-dpr2-1437   | disabled | 2   | 1437  |

GPU off is simulated with `--disable-gpu`. The odd width 1437 exists to catch sub-pixel rounding drift between even and odd container sizes.

## Invariants checked per cell

- **Zero page errors.** Any `pageerror` event fails the cell.
- **Console errors**, with one tolerated exception: a 404 for `viewer-config.json` is allowed (optional config file). Every other console error fails the cell.
- **Keyboard scrub lead 1..10.** The harness focuses `#lead-day`, presses ArrowRight one step at a time, and waits for the value to actually change after each press before pressing again (`pressAndSettle`). It does not fire keys on a fixed 50ms cadence because the viewer's own input handling drops keystrokes at that rate; per-press settling makes the scrub deterministic. Ending value must be exactly 10.
- **Layout probes vs `expectations.json`.** Each probe reads a bounding box property (e.g. `#lead-day` width, `.dock-playback` height, `.panel-field` width) and compares against the seeded expectation within an absolute pixel tolerance. Expectations live only in `qa/expectations.json`.
- **Warm back-scrub zero-network invariant.** After reaching lead 10, a request counter is attached (excluding `data:` URLs), then the harness scrubs ArrowLeft back down to lead 1, waits 250ms, and requires that zero network requests were issued. Once the page is warm, stepping through leads must be served entirely from memory/cache.

## Current known results (as of 2026-08-21)

- Seed PASS. `expectations.json` was re-baselined today following intentional layout changes (cover-fit map, legend chips).
- All 8/8 cells pass the keyboard scrub (lead reaches 10) and all layout probes within tolerance.
- All 8/8 cells FAIL the warm back-scrub invariant: each reports 4 identical remote requests during the measurement window, touching `insights.json`, the viewer manifest, `.zmetadata`, zarr chunk files, and `scores-summary.json`.
- Adjudication pending: these may be legitimate one-time lazy loads that happen to land inside the measurement window (in which case the invariant needs refining, not the app), or a real cache miss where already-fetched data is refetched on back-scrub. Not yet determined.

## TODO

- [ ] Adjudicate the warm back-scrub failure: distinguish one-time lazy loads inside the measurement window from a genuine cache miss, then fix either the app or the invariant accordingly.
- [ ] Add dateline seam pixel scan (screenshot-based dark-ratio check near the antimeridian seam; thresholds already stubbed in `expectations.json`).
- [ ] Flipbook frame count assertion (enforce minimum paints per lead step per panel).
- [ ] Paint latency budget checks against `budgets.warmLeadStepMs` / `coldLeadStepMs`.
- [ ] Em dash DOM scan (fail if any em dash sneaks into rendered viewer content).
- [ ] Exploratory random-walk mode: randomized lead sequences and interactions beyond the fixed 1..10 path, using `exploration.defaultSeconds`.
