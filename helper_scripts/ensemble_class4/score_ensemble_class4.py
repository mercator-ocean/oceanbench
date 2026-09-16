# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Score an ensemble challenger on the observation (Class IV) axis, one forecast start at a time.

The metrics, the matchup and the records all come from :mod:`oceanbench.core.ensemble_class4`
unchanged. This script only opens one forecast start of one challenger, hands it to the library
together with the reference observations of that start and the sigma artifact, and writes what
comes back. Nothing here computes a score.

Two files are written per start. The records file holds the per-start metric rows, which are
what a single start can say on its own. The rows file holds the finite matchup rows themselves,
observation by observation and member by member, because the published Class IV number pools the
observations of every start rather than averaging the per-start values, and that pooling needs the
rows. The aggregate step reads the rows back, rebuilds one matchup per variable spanning the whole
year and calls the same library function once, so the pooled value is the library's own and not a
weighted mean assembled here.

    python score_ensemble_class4.py --output-root DIR score --challenger glonet2-ens-icp \
        --start-date 2024-01-03
    python score_ensemble_class4.py --output-root DIR aggregate --challenger glonet2-ens-icp \
        --expect-starts 52
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import pandas
import s3fs
import xarray

from oceanbench.core.challenger_datasets import (
    _GLOENS_INITIALISATION_TO_FIRST_DAY,
    _open_gloens_forecast_week,
    _prepared_challenger_week_dataset,
)
from oceanbench.core.curvilinear_staging import GLOENS_SOURCE_NAME
from oceanbench.core.dataset_source import with_dataset_source
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.ensemble_class4 import (
    Class4EnsembleMatchup,
    SigmaLookup,
    ensemble_class4_matchup,
    ensemble_class4_rank_histograms,
    ensemble_class4_records,
)
from oceanbench.core.ensemble_gridded import ENSEMBLE_DIMENSION
from oceanbench.core.references.observations import observations as reference_observations
from oceanbench.core.score_records import RunContext, records_to_dataframe
from oceanbench.core.version import __version__ as OCEANBENCH_VERSION

CLOUDFERRO_ENDPOINT = "https://s3.waw3-1.cloudferro.com"
OCEANBENCH_BUCKET = "oceanbench-bucket"
ML_FORECAST_DEV_PREFIX = "dev/ml-forecast-outputs"

REFERENCE_NAME = "class4"
REGION_NAME = "global"
SCORED_YEAR = 2024

DEFAULT_SIGMA_STORE = "/scratch/jseillade/probax/campaign/artifacts/sigma-lookup-v3.0.0.zarr"

# Every variable the Class IV observation store carries. The depth bins each one is reported on,
# and the 15 m target the velocity components are read at, are the library's own.
SCORED_VARIABLES = (
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
    Variable.SEA_WATER_SALINITY,
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    Variable.EASTWARD_SEA_WATER_VELOCITY,
    Variable.NORTHWARD_SEA_WATER_VELOCITY,
)

# The ensemble fields of a glonet2 family store. Each of them has a ``*_control`` companion
# holding the single control forecast, which declares the same standard name as the ensemble
# field beside it, so a store handed to the standard-name rename whole gives two variables one
# name. The control is not part of the ensemble and is dropped here, as the gridded scorer drops
# it by naming the fields it reads.
CHALLENGER_ENSEMBLE_VARIABLE_NAMES = ("thetao", "so", "zos", "uo", "vo")

STORE_ML_FORECAST_DEV = "ml-forecast-dev"
STORE_GLOENS_WEEK = "gloens-week"


@dataclass(frozen=True)
class ChallengerSpecification:
    """Everything that differs between the ensemble challengers scored here."""

    name: str
    version: str
    store_layout: str
    member_dimension: str
    lead_days_count: int
    # Whether the challenger declares its own sea surface height basis to the Class IV seam. Only
    # GloEns does: it ships an inverse barometer and its own mean sea surface shift. A challenger
    # that declares none is given the reanalysis shift of -0.1148, which is the shift the glonet2
    # family was calibrated on.
    declares_dataset_source: bool


CHALLENGERS = {
    "glonet2-ens": ChallengerSpecification(
        name="glonet2-ens",
        version="glonet2-ens",
        store_layout=STORE_ML_FORECAST_DEV,
        member_dimension="member",
        lead_days_count=9,
        declares_dataset_source=False,
    ),
    "glonet2-ens-icp": ChallengerSpecification(
        name="glonet2-ens-icp",
        version="glonet2-ens-icp",
        store_layout=STORE_ML_FORECAST_DEV,
        member_dimension="member",
        lead_days_count=9,
        declares_dataset_source=False,
    ),
    "gloens": ChallengerSpecification(
        name="gloens",
        version="glo4-ens50_ng",
        store_layout=STORE_GLOENS_WEEK,
        member_dimension=ENSEMBLE_DIMENSION,
        lead_days_count=10,
        declares_dataset_source=True,
    ),
}

# The start date on the command line is the label the gridded campaign uses, so a start of one axis
# names the same forecast as the start of the other. For the glonet2 family that label is the first
# day the forecast predicts. For GloEns it is the initialisation the store is named after, whose
# first predicted day is the day after, which is the day the library reads the week from.
GLOENS_START_LABEL_TO_FIRST_DAY = _GLOENS_INITIALISATION_TO_FIRST_DAY


def _filesystem() -> s3fs.S3FileSystem:
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        raise RuntimeError(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set for the dev prefix; "
            "source the campaign credentials file"
        )
    return s3fs.S3FileSystem(key=key, secret=secret, client_kwargs={"endpoint_url": CLOUDFERRO_ENDPOINT})


def _open_dev_prefix_week(specification: ChallengerSpecification, start_label: pandas.Timestamp) -> xarray.Dataset:
    """One forecast start of a glonet2 family store, as the library's weekly challenger dataset.

    The store is named after the first day it predicts and holds that day at time index zero, so
    the time axis becomes the lead day index with no offset.
    """
    filesystem = _filesystem()
    root = f"{OCEANBENCH_BUCKET}/{ML_FORECAST_DEV_PREFIX}/{specification.name}/{start_label:%Y%m%d}.zarr"
    store = xarray.open_zarr(s3fs.S3Map(root=root, s3=filesystem, check=False), consolidated=True)
    ensemble_fields = [name for name in CHALLENGER_ENSEMBLE_VARIABLE_NAMES if name in store.data_vars]
    forecast_days = store[ensemble_fields].isel(time=slice(0, specification.lead_days_count))
    week = _prepared_challenger_week_dataset(forecast_days, f"{specification.name} challenger dataset open")
    return week.rename({specification.member_dimension: ENSEMBLE_DIMENSION})


def _open_challenger_start(
    specification: ChallengerSpecification, start_label: pandas.Timestamp
) -> tuple[xarray.Dataset, pandas.Timestamp]:
    """The challenger dataset of one start, shaped as the library reads it, and its first day."""
    if specification.store_layout == STORE_GLOENS_WEEK:
        first_day = start_label + GLOENS_START_LABEL_TO_FIRST_DAY
        week = _open_gloens_forecast_week(first_day.to_pydatetime())
    else:
        first_day = start_label
        week = _open_dev_prefix_week(specification, start_label)
    challenger = week.expand_dims({Dimension.FIRST_DAY_DATETIME.key(): [first_day.to_datetime64()]})
    if specification.declares_dataset_source:
        challenger = with_dataset_source(challenger, kind="challenger", name=GLOENS_SOURCE_NAME)
    return challenger, first_day


def _run_context(specification: ChallengerSpecification) -> RunContext:
    return RunContext(
        challenger=specification.name,
        challenger_version=specification.version,
        year=SCORED_YEAR,
        region=REGION_NAME,
        oceanbench_version=OCEANBENCH_VERSION,
    )


def _member_column_names(member_count: int) -> list[str]:
    return [f"member_{member_index:02d}" for member_index in range(member_count)]


def _matchup_rows_frame(matchup: Class4EnsembleMatchup) -> pandas.DataFrame:
    """The matchup as one flat frame, so the aggregate can rebuild it without recomputing it."""
    member_columns = dict(zip(_member_column_names(matchup.member_count), matchup.member_values.T, strict=True))
    return matchup.observations.assign(**member_columns)


def _matchup_from_rows_frame(variable: str, rows: pandas.DataFrame) -> Class4EnsembleMatchup:
    member_columns = [name for name in rows.columns if name.startswith("member_")]
    member_values = rows[sorted(member_columns)].to_numpy("float64")
    observations = rows.drop(columns=member_columns).reset_index(drop=True)
    return Class4EnsembleMatchup(variable, observations, member_values)


def _rank_histogram_frame(histograms: dict, specification: ChallengerSpecification) -> pandas.DataFrame:
    rows = [
        {
            "challenger": specification.name,
            "variable": variable,
            "depth": depth_bin,
            "lead_day": lead_day,
            "dressing_mode": mode,
            "rank_bin": bin_index,
            "frequency": float(frequency),
        }
        for (variable, depth_bin, lead_day, mode), histogram in sorted(histograms.items())
        for bin_index, frequency in enumerate(histogram)
    ]
    return pandas.DataFrame(rows)


def _start_directory(output_root: Path, challenger_key: str) -> Path:
    return output_root / challenger_key / "per-start"


def _rows_path(output_root: Path, challenger_key: str, start_label: pandas.Timestamp, variable: str) -> Path:
    return _start_directory(output_root, challenger_key) / f"rows-{start_label:%Y%m%d}-{variable}.parquet"


def _records_path(output_root: Path, challenger_key: str, start_label: pandas.Timestamp) -> Path:
    return _start_directory(output_root, challenger_key) / f"records-{start_label:%Y%m%d}.parquet"


def _ranks_path(output_root: Path, challenger_key: str, start_label: pandas.Timestamp) -> Path:
    return _start_directory(output_root, challenger_key) / f"ranks-{start_label:%Y%m%d}.parquet"


def _sigma_lookup(sigma_store: str | None) -> SigmaLookup | None:
    return None if sigma_store in (None, "none") else SigmaLookup(sigma_store)


def _score_command(arguments: argparse.Namespace) -> None:
    specification = CHALLENGERS[arguments.challenger]
    start_label = pandas.Timestamp(arguments.start_date)
    output_root = Path(arguments.output_root)
    _start_directory(output_root, arguments.challenger).mkdir(parents=True, exist_ok=True)

    challenger, first_day = _open_challenger_start(specification, start_label)
    if arguments.member_limit is not None:
        challenger = challenger.isel({ENSEMBLE_DIMENSION: slice(0, arguments.member_limit)})
    print(f"challenger={arguments.challenger} start={start_label:%Y-%m-%d} first_day={first_day:%Y-%m-%d}")
    print(
        f"members={challenger.sizes[ENSEMBLE_DIMENSION]} lead_days={challenger.sizes[Dimension.LEAD_DAY_INDEX.key()]}"
    )

    observations_dataset = reference_observations(challenger)
    matchups = ensemble_class4_matchup(challenger, observations_dataset, SCORED_VARIABLES)
    print(f"matched variables={[matchup.variable for matchup in matchups]}")

    for matchup in matchups:
        rows = _matchup_rows_frame(matchup)
        rows.to_parquet(_rows_path(output_root, arguments.challenger, start_label, matchup.variable), index=False)
        print(f"rows variable={matchup.variable} count={len(rows)} members={matchup.member_count}")

    sigma_lookup = _sigma_lookup(arguments.sigma_store)
    records = ensemble_class4_records(
        matchups,
        context=_run_context(specification),
        reference=REFERENCE_NAME,
        sigma_lookup=sigma_lookup,
    )
    dataframe = records_to_dataframe(records)
    per_start = dataframe[dataframe["start_date"].notna()]
    per_start.to_parquet(_records_path(output_root, arguments.challenger, start_label), index=False)
    print(f"records={len(per_start)}")

    histograms = ensemble_class4_rank_histograms(matchups, sigma_lookup=sigma_lookup)
    _rank_histogram_frame(histograms, specification).to_parquet(
        _ranks_path(output_root, arguments.challenger, start_label), index=False
    )
    print(f"rank_histograms={len(histograms)}")


def _rows_paths_by_variable(output_root: Path, challenger_key: str) -> dict[str, list[Path]]:
    paths_by_variable: dict[str, list[Path]] = {}
    for path in sorted(_start_directory(output_root, challenger_key).glob("rows-*.parquet")):
        variable = path.stem.split("-", 2)[2]
        paths_by_variable.setdefault(variable, []).append(path)
    return paths_by_variable


def _refuse_incomplete_year(paths: list[Path], variable: str, expected_starts: int) -> None:
    if len(paths) != expected_starts:
        raise SystemExit(
            f"variable {variable} has {len(paths)} scored starts, expected {expected_starts}: "
            "the pooled Class IV value would be taken over an incomplete year"
        )


def _aggregate_command(arguments: argparse.Namespace) -> None:
    specification = CHALLENGERS[arguments.challenger]
    output_root = Path(arguments.output_root)
    paths_by_variable = _rows_paths_by_variable(output_root, arguments.challenger)
    if not paths_by_variable:
        raise SystemExit(f"no scored starts found under {_start_directory(output_root, arguments.challenger)}")

    sigma_lookup = _sigma_lookup(arguments.sigma_store)
    context = _run_context(specification)
    record_frames = []
    histogram_frames = []
    for variable, paths in sorted(paths_by_variable.items()):
        _refuse_incomplete_year(paths, variable, arguments.expect_starts)
        rows = pandas.concat([pandas.read_parquet(path) for path in paths], ignore_index=True)
        matchup = _matchup_from_rows_frame(variable, rows)
        print(f"variable={variable} starts={len(paths)} rows={len(rows)} members={matchup.member_count}")
        record_frames.append(
            records_to_dataframe(
                ensemble_class4_records([matchup], context=context, reference=REFERENCE_NAME, sigma_lookup=sigma_lookup)
            )
        )
        histogram_frames.append(
            _rank_histogram_frame(ensemble_class4_rank_histograms([matchup], sigma_lookup=sigma_lookup), specification)
        )

    challenger_root = output_root / arguments.challenger
    scores = pandas.concat(record_frames, ignore_index=True)
    scores.to_parquet(challenger_root / "scores.parquet", index=False)
    pandas.concat(histogram_frames, ignore_index=True).to_parquet(
        challenger_root / "rank-histograms.parquet", index=False
    )
    pooled = scores[scores["start_date"].isna()]
    print(f"wrote {challenger_root / 'scores.parquet'} records={len(scores)} pooled={len(pooled)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--challenger", required=True, choices=sorted(CHALLENGERS))
    score_parser.add_argument("--start-date", required=True)
    score_parser.add_argument("--sigma-store", default=DEFAULT_SIGMA_STORE)
    score_parser.add_argument("--member-limit", type=int, default=None)
    score_parser.set_defaults(handler=_score_command)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--challenger", required=True, choices=sorted(CHALLENGERS))
    aggregate_parser.add_argument("--expect-starts", type=int, required=True)
    aggregate_parser.add_argument("--sigma-store", default=DEFAULT_SIGMA_STORE)
    aggregate_parser.set_defaults(handler=_aggregate_command)

    arguments = parser.parse_args()
    arguments.handler(arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
