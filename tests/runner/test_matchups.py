# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""The Class-4 match-up artifact must reproduce the Class-4 metric exactly.

The synthetic test proves that reshaping an observation-with-model frame into the
match-up artifact and recomputing the RMSD from it equals the untouched core
reduction (``compute_class4_rmsd_table``) cell-for-cell. The real test (skipped
unless the parity stage cache is present) proves the same against the published
metric formatter over glonet_1_degree/global observations.
"""

import os

import numpy
import pandas
import pytest

from oceanbench.core.classIV_support import compute_class4_rmsd_table
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.runner import matchups
from oceanbench.runner.records import RunContext

_STAGE_DIRECTORY = os.environ.get("OCEANBENCH_STAGE_DIR")


def _context() -> RunContext:
    return RunContext(
        challenger="glonet_1_degree",
        challenger_version="0.2.1",
        year=2024,
        region="global",
        oceanbench_version="0.2.1",
    )


def _synthetic_observation_frame(seed: int, variable_key: str) -> pandas.DataFrame:
    generator = numpy.random.default_rng(seed)
    first_days = numpy.array(["2024-01-03", "2024-01-10", "2024-01-17"], dtype="datetime64[ns]")
    depth_bins = (
        ["surface", "0-5m", "100-300m"]
        if variable_key != Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
        else ["surface"]
    )
    rows = []
    for first_day in first_days:
        for depth_bin in depth_bins:
            for lead_day in (0, 3, 9):
                for _ in range(int(generator.integers(1, 6))):
                    observation_value = float(generator.normal())
                    rows.append(
                        {
                            Dimension.TIME.key(): pandas.Timestamp(first_day) + pandas.Timedelta(days=lead_day),
                            Dimension.LATITUDE.key(): float(generator.uniform(-60, 60)),
                            Dimension.LONGITUDE.key(): float(generator.uniform(-180, 180)),
                            "first_day": first_day,
                            Dimension.DEPTH.key(): float(generator.uniform(0, 300)),
                            "lead_day": lead_day,
                            "observation_value": observation_value,
                            "depth_bin": depth_bin,
                            "model_value": observation_value + float(generator.normal(scale=0.4)),
                        }
                    )
    return pandas.DataFrame(rows)


def test_recomputed_rmsd_equals_core_reduction_synthetic():
    context = _context()
    variables = [
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
        Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
        Variable.SEA_WATER_SALINITY,
    ]
    artifact_parts = []
    core_parts = []
    for seed, variable in enumerate(variables):
        frame = _synthetic_observation_frame(seed, variable.key())
        artifact_parts.append(matchups._shaped_matchups(frame, variable.key(), context))
        core_parts.append(compute_class4_rmsd_table(frame, variable.key()))

    artifact = pandas.concat(artifact_parts, ignore_index=True)
    recomputed = matchups.recompute_class4_rmsd(artifact)
    # The artifact stores lead_day 1-based; the core table is 0-based.
    recomputed = recomputed.assign(lead_day=recomputed["lead_day"] - 1)
    core = pandas.concat(core_parts, ignore_index=True)

    keys = ["variable", "depth_bin", "lead_day"]
    merged = core.sort_values(keys).merge(recomputed.sort_values(keys), on=keys, suffixes=("_core", "_artifact"))
    assert len(merged) == len(core) == len(recomputed)
    numpy.testing.assert_allclose(
        merged["rmsd_artifact"].to_numpy(), merged["rmsd_core"].to_numpy(), rtol=0.0, atol=1e-12
    )
    assert (merged["count_core"] == merged["count_artifact"]).all()


def test_shaped_matchups_columns_and_sla_shift():
    context = _context()
    frame = _synthetic_observation_frame(0, Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key())
    shaped = matchups._shaped_matchups(frame, Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(), context)
    assert list(shaped.columns) == matchups.MATCHUP_COLUMNS
    assert (shaped["lead_day"] >= 1).all()
    assert numpy.allclose(shaped["sla_shift"], -0.1148)

    temperature = matchups._shaped_matchups(
        _synthetic_observation_frame(1, Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()),
        Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
        context,
    )
    assert temperature["sla_shift"].isna().all()


def test_write_class4_matchups_roundtrips(tmp_path):
    context = _context()
    frame = _synthetic_observation_frame(0, Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key())
    artifact = matchups._shaped_matchups(frame, Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(), context)
    path, size = matchups.write_class4_matchups(artifact, str(tmp_path / "class4-matchups.parquet"))
    assert size > 0
    reloaded = pandas.read_parquet(path)
    pandas.testing.assert_frame_equal(reloaded, artifact, check_dtype=False)


@pytest.mark.skipif(
    _STAGE_DIRECTORY is None or not os.path.isdir(_STAGE_DIRECTORY),
    reason="parity stage cache (OCEANBENCH_STAGE_DIR) not available",
)
def test_real_matchups_reproduce_class4_metric():
    import re

    from oceanbench.core.classIV import rmsd_class4_validation
    from oceanbench.core.references.observations import observations
    from oceanbench.core.regions import subset_dataset_to_region
    import oceanbench.datasets.challenger as challenger_datasets

    # The small IBI box keeps the observation count low enough for a unit test while
    # still exercising the full open -> interpolate -> reshape -> recompute pipeline.
    variables = [Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID, Variable.SEA_WATER_POTENTIAL_TEMPERATURE]
    challenger = subset_dataset_to_region(challenger_datasets.glonet_1_degree(), "ibi")
    observation_dataset = subset_dataset_to_region(observations(challenger), "ibi")

    artifact = matchups.class4_matchups(challenger, observation_dataset, variables, context=_context())
    recomputed = matchups.recompute_class4_rmsd(artifact)

    metric = rmsd_class4_validation(challenger, observation_dataset, variables)
    for _, row in recomputed.iterrows():
        label = next(index for index in metric.index if f"[{row['variable']}]{{{row['depth_bin']}}}" in index)
        column = next(column for column in metric.columns if re.search(rf"\b{int(row['lead_day'])}\s*$", str(column)))
        assert row["rmsd"] == pytest.approx(float(metric.loc[label, column]), abs=1e-9)
