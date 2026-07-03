<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Score page prototype (v0)

Self-contained static score page — the v0 skeleton the real site grows from at cutover.
It is deliberately separate from the existing `website/` Quarto tree.

## What it is

A **no-ranking scorecard** (contracts.md §1): rows are challengers in neutral order
(baselines pinned to the top), columns are key metrics at a selectable reference lead day
(1/3/5/7/10). Cells show the mean over forecast starts with the bootstrap 95% confidence
interval. Any column is sortable on click, but there is **no default rank order, no composite
score, and no verdict colouring**. Below the table, a plain-language summary card per
challenger reads the same numbers at a non-expert level.

## Data path

- `scores.parquet` is the **canonical artifact**. It is read directly in the browser with the
  vendored [hyparquet](https://hyperparam.app) reader (MIT, `vendor/hyparquet/`, no CDN). The
  per-start means shown in the table are aggregated client-side from it.
- `scores-summary.json` carries the **precomputed** bootstrap confidence intervals and
  skill-vs-baseline (emitted next to the parquet by `oceanbench.publish.benchmark.publish_scores`),
  because recomputing a 1000-draw paired bootstrap in the browser is wasteful. The parquet
  remains the source of truth; the summary is a derived convenience.
- `challengers.json` (optional) supplies display names and the `is_baseline` flag used to pin
  baselines. Without it, slugs are shown and no rows are pinned. It is the in-repo, versioned
  challenger registry (`challengers.json` at the repository root, schema-validated against
  `schemas/challengers.schema.json`); `oceanbench.publish.publish_challengers_registry` copies it
  into the catalog root next to `scores.parquet`, and the catalog's `challengers_url` points at it.

## Running locally

The `data/` directory is generated (git-ignored). Populate it from a publish run, then serve:

```sh
python -m oceanbench.publish...   # writes scores.parquet + scores-summary.json
# emit the challenger registry (display names + is_baseline) next to the scores:
python -c "from oceanbench.publish import publish_challengers_registry; publish_challengers_registry('<output_root>')"
cp <output_root>/scores.parquet <output_root>/scores-summary.json <output_root>/challengers.json website-rebuild/scores/data/
python -m http.server -d website-rebuild/scores 8000
# open http://localhost:8000/
```

## Status flags

- The challenger registry now carries the five official models (and their 1° variants) plus the
  `climatology` and `persistence` baselines (`is_baseline: true`), so baseline rows are pinned to
  the top once their scores land. The skill columns stay hidden until the summary carries
  `skill_vs_*`; that path is already coded and switches on automatically.
