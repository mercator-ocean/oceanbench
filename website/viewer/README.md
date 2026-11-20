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
- It comes with its data builder and links from the score tables to it.

## Where the data comes from

- Files on CloudFerro, under `dev/benchmark/rebuild-preview/viewer/data/` (262 GB).
- Made with OceanBench 0.5.0, not 0.6.0, so some numbers differ from the score tables.
  Example: GLONET 15 m zonal current at lead 1 is 0.12 in the table and 0.204 here.
- Its eddies come from the detector of 0.5.0, not from py-eddy-tracker, so the eddy
  method note describes them only once the data is rebuilt.
- Only the year 2024.
- To rebuild with the release that merges this branch.

## Adding a model

With the `viewer` extra installed (`pip install "oceanbench[viewer]"`), run
`oceanbench evaluate <challenger>.py --viewer-artifacts-dir <directory>` for each
region, see [Building the viewer data](https://oceanbench.edito.eu/documentation/viewer-data/). The index files (`datasets.json`, `insights.json`,
`scores-summary.json`) are merged automatically. The bands of the lead-day chart come
from a bootstrap over the start dates, those of the start-date chart from a block bootstrap
over 5 degree boxes of observations (2 degree for IBI).

Cost, measured on GLONET (1/4 degree, global) with two start dates on 16 cores: about
5 min of scores, 11 min of viewer data and 0.6 GB of files. A year of 52 start dates is
about 15 GB for GLONET; a 1/12 degree model has nine times as many map cells. The eddy
census needs py-eddy-tracker (see the documentation) and takes about 35 s of CPU per start
date and lead day, run in 8 processes by default. Details under "Costs" in the
documentation.

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
