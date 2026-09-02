# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Self-contained overlay scorecard for local evaluation (contracts.md §7).

Renders the same no-ranking scorecard semantics as the website scores page
(``website-rebuild/scores/``): mean +/- 95% CI over forecast starts, baselines pinned,
neutral order, no composite score, but with the user's model overlaid on the published
challengers and clearly labelled "your model".

Data is **inlined** into ``index.html`` rather than fetched: a local file opened over
``file://`` cannot ``fetch()`` sibling files nor load an ES module (both are blocked by the
browser same-origin policy for ``file://``), which is exactly what the website's
``app.js`` relies on. Inlining the pre-aggregated cells and a classic (non-module) renderer
makes the report open with a plain double-click, no server required. The aggregation itself
reuses :func:`oceanbench.publish.aggregate.aggregate_scores`, so the numbers are identical
to the hosted page.
"""

import json
from pathlib import Path

import pandas

from oceanbench.publish.aggregate import aggregate_scores, summary_to_json_records

INDEX_FILENAME = "index.html"
_ASSETS_DIRECTORY = Path(__file__).parent / "assets"

YOUR_MODEL_SLUG = "your_model"


def _asset(name: str) -> str:
    return (_ASSETS_DIRECTORY / name).read_text(encoding="utf-8")


def _shared_start_published(published: pandas.DataFrame, your_model: pandas.DataFrame) -> pandas.DataFrame:
    """Restrict published records to the forecast starts the user's model covers.

    The overlay is a like-for-like comparison: both are aggregated over the same starts so the
    published challenger and the user's model coincide exactly when the user re-scores that
    challenger. Restricting published records to the local forecast starts avoids comparing
    means over different samples.
    """
    your_starts = pandas.to_datetime(your_model["start_date"].dropna().unique())
    published_dates = pandas.to_datetime(published["start_date"])
    return published[published_dates.isin(your_starts).to_numpy()].reset_index(drop=True)


def _aggregated_cells(scores: pandas.DataFrame) -> list[dict]:
    summary = aggregate_scores(scores)
    return summary_to_json_records(summary)


def _challenger_metadata(
    your_model_scores: pandas.DataFrame,
    published_scores: pandas.DataFrame | None,
    published_challengers: dict | None,
) -> dict:
    metadata: dict[str, dict] = {}
    if published_challengers:
        for slug, entry in published_challengers.items():
            metadata[slug] = {
                "display_name": entry.get("display_name", slug),
                "is_baseline": bool(entry.get("is_baseline", False)),
                "is_your_model": False,
            }
    if published_scores is not None:
        for slug in published_scores["challenger"].dropna().unique():
            metadata.setdefault(str(slug), {"display_name": str(slug), "is_baseline": False, "is_your_model": False})
    metadata[YOUR_MODEL_SLUG] = {
        "display_name": "Your model",
        "is_baseline": False,
        "is_your_model": True,
    }
    return metadata


def build_scorecard_payload(
    your_model_scores: pandas.DataFrame,
    published_scores: pandas.DataFrame | None,
    published_challengers: dict | None,
    *,
    region: str,
    year: int,
    generated_at: str,
) -> dict:
    """Assemble the inlined scorecard payload (aggregated cells + challenger metadata)."""
    frames = [your_model_scores]
    if published_scores is not None and not published_scores.empty:
        frames.append(_shared_start_published(published_scores, your_model_scores))
    combined = pandas.concat(frames, ignore_index=True)
    # Only per-start metrics have a start distribution to aggregate (realism records carry a null
    # start_date, contracts.md §3.2); the scorecard shows the mean +/- CI per-start metrics.
    combined = combined[combined["start_date"].notna()].reset_index(drop=True)
    return {
        "generated_at": generated_at,
        "region": region,
        "year": year,
        "your_model_slug": YOUR_MODEL_SLUG,
        "cells": _aggregated_cells(combined),
        "challengers": _challenger_metadata(your_model_scores, published_scores, published_challengers),
    }


def write_overlay_scorecard(
    output_directory: Path,
    *,
    your_model_scores: pandas.DataFrame,
    published_scores: pandas.DataFrame | None,
    published_challengers: dict | None,
    region: str,
    year: int,
    generated_at: str,
) -> str:
    """Write a self-contained ``index.html`` overlay scorecard; return its path."""
    output_directory.mkdir(parents=True, exist_ok=True)
    payload = build_scorecard_payload(
        your_model_scores,
        published_scores,
        published_challengers,
        region=region,
        year=year,
        generated_at=generated_at,
    )
    index_path = output_directory / INDEX_FILENAME
    index_path.write_text(_render_html(payload), encoding="utf-8")
    return str(index_path)


def _render_html(payload: dict) -> str:
    data_json = json.dumps(payload, sort_keys=True)
    styles = _asset("scorecard.css")
    script = _asset("scorecard.js")
    return f"""<!DOCTYPE html>
<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>OceanBench: local evaluation scorecard</title>
<style>{styles}</style>
</head>
<body>
<header class="page-header">
  <h1>OceanBench: local evaluation</h1>
  <p class="tagline"><strong>Your model</strong> overlaid on the published challengers.
  Mean &plusmn; 95% CI over the shared forecast starts. No ranking, no composite score:
  baselines are pinned, order is neutral.</p>
</header>
<section class="controls" id="controls"></section>
<p class="status" id="status">Loading\u2026</p>
<main id="main" hidden>
  <h2>Scorecard</h2>
  <p class="section-note" id="scorecard-note"></p>
  <div class="table-scroll"><table id="scorecard"><thead></thead><tbody></tbody></table></div>
  <p class="legend">The <span class="your-model-chip">your model</span> row is highlighted.
  Published means are aggregated over the same starts as your model, so a re-scored published
  challenger coincides with your model exactly.</p>
</main>
<footer class="page-footer">
  <p id="provenance"></p>
  <p>Generated using E.U. Copernicus Marine Service Information;
  https://doi.org/10.48670/moi-00021 ; https://doi.org/10.48670/moi-00016. OceanBench-generated
  derived product, not the authoritative Copernicus Marine product.</p>
</footer>
<script type="application/json" id="scorecard-data">{data_json}</script>
<script>{script}</script>
</body>
</html>
"""
