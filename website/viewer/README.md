<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# OceanBench viewer

A map viewer for OceanBench forecasts. Pick a model, a variable, a start date and a lead
day to see the forecast map. You can compare two models (side by side, swipe or
difference), show errors against observations, animate currents and click for a depth
profile. Every view is saved in the URL, so a link reopens the same view.

It is a static page in plain JavaScript, with no build step and no server. It computes no
scores: it only displays files prepared in advance.

## Where to see it

- Dev page: https://oceanbench-viewer-dev.lab.dive.edito.eu/viewer.html
- Branch `viewer-dev` (PR #367) is main plus scoring without notebooks, this viewer, its
  data builder and links from the score tables to it.
- The page updates itself about 20 minutes after a push to `viewer-dev`. To force an
  update: GitHub Actions > "Update website" > branch `viewer-dev`.

## Where the data comes from

- Files on CloudFerro, under `dev/benchmark/rebuild-preview/viewer/data/` (262 GB).
- Made with OceanBench 0.5.0, not 0.6.0, so some numbers differ from the score tables.
  Example: GLONET 15 m zonal current at lead 1 is 0.12 in the table and 0.204 here.
- Only the year 2024.
- To rebuild with the release that merges this branch.

## Adding a model

Run `oceanbench evaluate <challenger>.py --viewer-artifacts-dir <directory>` for each
region, see `docs/viewer-data.rst`. The index files (`datasets.json`, `insights.json`,
`scores-summary.json`) are merged automatically, and the confidence bands come from a
bootstrap over the start dates. Cost for one year (52 start dates) on one node with
120-180 GB of RAM:

| Model grid  | Time                                          | Storage      |
| ----------- | --------------------------------------------- | ------------ |
| 1/12 degree | about 3 h, about 20 h in practice (downloads) | 60-70 GB     |
| 1/4 degree  | about 3 h                                     |              |
| 1 degree    | 2 to 10 h                                     | about 6.5 GB |
| IBI         | 20 min to 2 h                                 |              |

A new year means rerunning every model and adding a year selector (the page assumes 2024).

## Running it locally

```sh
python -m http.server -d website/viewer 8799
# open http://127.0.0.1:8799/
```

It reads the CloudFerro data by default. Add `?data=<url>` to use other data, or
`?data=local` for `website/viewer/data/`.

On the website, `website/viewer.qmd` shows the viewer in an iframe. Quarto copies this
folder only on `quarto render`, so `quarto preview` does not pick up viewer edits.

## Files

- `index.html` and `app.js`: the page and most of the logic.
- `config.js`: where the data is read from. The score page (`website/interactive-scores.js`)
  uses it too, to know which models get a viewer link.
- `modules/`: data reading, charts, overlays and other helpers.
- `state/`, `styles/`: shared view state and CSS.
- `vendor/`: third-party code. `qa/`: tests.

## Tests

```sh
cd website/viewer && node qa/run.mjs
```

Opens the page in a headless browser (Playwright) and checks behaviour, layout and drawn
pixels against saved references. Update the references only when the drawing changed on
purpose (see `qa/README.md`).

## Known issues

- The score tables redraw once the viewer list loads, which resets horizontal scroll on
  phones.
- Score cell links add many Tab stops for keyboard users.
- The thin line at 180 degrees on GLONET maps is in GLONET's data, not a viewer bug.
