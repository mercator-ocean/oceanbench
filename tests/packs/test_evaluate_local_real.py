# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Real-data proof of the local-evaluation overlay (contracts.md §7), skipped by default.

Builds a demo quick pack from the staged 1-degree references, materialises glonet_1_degree's own
forecasts as the "user model", scores them against the pack and asserts the per-start values match
the published glonet_1_degree scores.parquet exactly — the overlay's agreement claim, on real data.

Needs the warm stage; enable with::

    OCEANBENCH_RUN_PACK_STAGE_TESTS=1 OCEANBENCH_STAGE_DIR=/path/to/stage \\
      pytest tests/packs/test_evaluate_local_real.py
"""

import os
from pathlib import Path

import pandas
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("OCEANBENCH_RUN_PACK_STAGE_TESTS") != "1" or not os.environ.get("OCEANBENCH_STAGE_DIR"),
    reason="Real-data pack test; set OCEANBENCH_RUN_PACK_STAGE_TESTS=1 and OCEANBENCH_STAGE_DIR to run.",
)

_PUBLISHED_SCORES = "website-rebuild/scores/data/scores.parquet"
_PUBLISHED_CHALLENGERS = "website-rebuild/scores/data/challengers.json"
_START_LIMIT = 2


def _materialise_forecast(dataset, path: Path):
    loaded = dataset.load()
    for variable_name in loaded.variables:
        loaded[variable_name].encoding.pop("chunks", None)
    loaded.to_zarr(str(path), mode="w", consolidated=True)


def test_glonet_1_degree_local_evaluation_matches_published(tmp_path):
    from oceanbench.core import challenger_datasets
    from oceanbench.core.dataset_utils import Dimension
    from oceanbench.core.runtime_configuration import RuntimeConfiguration, set_runtime_configuration
    from oceanbench.packs.builder import PackSources, build_pack
    from oceanbench.packs.evaluate import evaluate_local, per_start_agreement

    stage_directory = os.environ["OCEANBENCH_STAGE_DIR"]
    set_runtime_configuration(RuntimeConfiguration(staged_components=("all",), stage_directory=stage_directory))

    pack_directory = tmp_path / "pack-quick-2024"
    build_pack(
        "quick",
        2024,
        PackSources(template_challenger="glonet_1_degree", start_limit=_START_LIMIT, region="global"),
        str(pack_directory),
    )

    forecast_path = tmp_path / "user-forecast.zarr"
    forecast = challenger_datasets.glonet_1_degree().isel({Dimension.FIRST_DAY_DATETIME.key(): slice(0, _START_LIMIT)})
    _materialise_forecast(forecast, forecast_path)

    result = evaluate_local(
        str(forecast_path),
        pack_directory=str(pack_directory),
        output_directory=str(tmp_path / "out"),
        year=2024,
        region="global",
        published_scores_path=_PUBLISHED_SCORES,
        published_challengers_path=_PUBLISHED_CHALLENGERS,
        starts_limit=_START_LIMIT,
        with_lagrangian=False,
        include_class4=False,
        include_realism=False,
    )

    published = pandas.read_parquet(_PUBLISHED_SCORES)
    agreement = per_start_agreement(result.scores, published)
    assert not agreement.empty
    assert agreement["absolute_difference"].max() < 1e-9
    assert Path(result.scorecard_path).exists()
