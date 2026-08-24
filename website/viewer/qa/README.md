# Viewer QA Harness

Permanent regression harness for the static viewer (`website/viewer/`). It boots a local static server, drives the real page in headless Chromium under Playwright, and asserts a fixed set of behavioral, layout and render invariants. It exists so that regressions are caught mechanically instead of by eye, and so that a refactor can be shown to have changed nothing the user sees.

## Running

```sh
node qa/run.mjs
```

Runs the 8 matrix cells sequentially, then the behavioural pass, and exits non-zero if anything reports failures.

```sh
node qa/run.mjs --seed
```

Re-baselines the layout probes in `expectations.json`: it loads the page once at the reference viewport (1440x900, dpr 1), measures each probe, and rewrites the `expected` values. Use this ONLY after an INTENTIONAL layout change. Do not widen tolerances to force green runs, and do not reseed to make an accidental regression pass.

```sh
node qa/run.mjs --seed-fingerprints
```

Re-baselines `fingerprints.json`, the frozen render fingerprints. Same rule, more strictly: a refactor must never reseed. Reseed only when the drawing changed on purpose, and say which states moved and why.

Requires `playwright` (with Chromium installed) and `python3` for the static file server. The server binds to a random free port on 127.0.0.1 and serves the `website/` directory; the page under test is `/viewer/index.html?qa=1`. The `qa` query parameter makes the app publish `window.oceanbenchViewerQaProbe`, which is how the colour-range probe reads the bounds the map was actually drawn with.

The fingerprints depend on the published viewer data at the configured data root, so republishing that data moves them too.

## Matrix

Six rendering-configuration cells and two behavioural cells (height always 900):

| Cell              | GPU      | DPR | Width | Opening state         |
| ----------------- | -------- | --- | ----- | --------------------- |
| gpu-dpr1-1440     | enabled  | 1   | 1440  | default               |
| gpu-dpr1-1437     | enabled  | 1   | 1437  | default               |
| gpu-dpr2-1440     | enabled  | 2   | 1440  | default               |
| nogpu-dpr1-1440   | disabled | 1   | 1440  | default               |
| nogpu-dpr2-1440   | disabled | 2   | 1440  | default               |
| nogpu-dpr2-1437   | disabled | 2   | 1437  | default               |
| behaviour-swipe   | enabled  | 1   | 1440  | 2 forecasts, swipe    |
| behaviour-overlay | enabled  | 1   | 1440  | Class-4 obs overlay   |

GPU off is simulated with `--disable-gpu`. The odd width 1437 exists to catch sub-pixel rounding drift between even and odd container sizes. The matrix used to spend all eight cells on device pixel ratio crossed with viewport width, which bought four ways to find the same rounding drift; two of those were traded for the behavioural cells at the same runtime. The swipe cell skips the layout probes, because two panels compose into one canvas there and the single-panel width expectation does not describe it.

## Invariants checked per cell

- **Zero page errors.** Any `pageerror` event fails the cell.
- **Console errors**, with one tolerated exception: a 404 for `viewer-config.json` is allowed (optional config file). Every other console error fails the cell.
- **Keyboard scrub lead 1..10.** The harness focuses `#lead-day`, presses ArrowRight one step at a time, and waits for the value to actually change after each press before pressing again (`pressAndSettle`). It does not fire keys on a fixed 50ms cadence because the viewer's own input handling drops keystrokes at that rate; per-press settling makes the scrub deterministic. Ending value must be exactly 10.
- **Layout probes vs `expectations.json`.** Each probe reads a bounding box property (`#lead-day` width, `.dock-playback` height, `.panel-field` width) and compares against the seeded expectation within an absolute pixel tolerance.
- **Classified warm back-scrub budget.** After reaching lead 10 and letting the forward scrub's tail finish, requests are counted per class (field chunk, match-up parquet, manifest, insight JSON, other) while the harness scrubs back down to lead 1. Budgets live in `expectations.json`. The point of the invariant is that a warm cache serves a back-scrub without re-reading field data; the counts are printed on every run whether or not they fail, so the numbers are visible rather than merely thresholded.
- **No em dashes in rendered text.** A house rule no linter can see, because it is about the DOM the user reads.

## The behavioural pass

Runs once per invocation, in one browser context at 1440x900 dpr 1, after the matrix.

- **Render fingerprints (P1).** Twelve viewer states, each opened by URL hash, each read back as an FNV-1a hash of every panel's field and overlay canvas, compared byte for byte against `fingerprints.json`. The states cover the three panel display modes, both map scopes, both year metrics, both overlay renderers, a regional zoom and the velocity path. Sampling starts only after the app has laid a panel out and drawn into it (a blank canvas is perfectly stable, so stability alone is not readiness), then waits for the fingerprint to repeat several samples running.
- **Fast-scrub race (P2).** Nine ArrowRight presses with no settling between them, then assert the page lands on exactly the lead-10 baseline. `pressAndSettle` is designed to avoid the render-token race; this fires into it on purpose.
- **Colour-range monotonicity (P3).** Sweeps lead 1 to 10 and requires the drawn colour range to grow and never shrink while the selection signature is unchanged. This is the grow-only property in `modules/stable-ranges.js`, which is otherwise purely visual and throws no errors when it breaks.
- **Particle liveness (P5).** With playback on, two samples 600ms apart must differ; with playback off, they must be identical. Catches the premultiplied-alpha fade stall documented in `modules/particles.js`.
- **Software inflate equivalence (P6).** One state re-rendered with `DecompressionStream` deleted, forcing the software DEFLATE decoder in `modules/zarr.js`, and required to produce the same pixels as the native path.

## Known behaviour

- As of 2026-08-24 the whole harness is green: 8/8 cells and the behavioural pass.
- The warm back-scrub budget is zero in every class and is met, so the "adjudication pending" note that used to live here is resolved: the four requests it recorded were the old counter's regex catching one-time lazy loads, not refetched field data.
- The fingerprint probe was negative-controlled by changing the difference colormap: it failed exactly the two states that draw with it and no others.

## Not covered

- Hover readouts, tooltips, trajectory seeding and the water-column click path are exercised by no probe.
- The rail charts are checked only through "no page error"; their SVG content is not asserted.
- Paint latency is not budgeted.
