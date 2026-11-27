# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Evaluate a challenger year as several jobs, each on a subset of the start dates (a shard), then merge them.

``evaluate-shard`` runs ``oceanbench evaluate`` with the viewer data on the start dates of one shard, without the
maps (they are built once over the year by the merge), and keeps what the merge needs in a directory: the gridded
score tables of each start date, the Lagrangian deviations of each start date and the Class IV match-ups.

``merge`` opens the challenger with all the start dates of the shards and writes the scores file and the viewer data
of the year, as one evaluation of all the start dates would:

- gridded RMSD: the mean over the start dates of the RMSD of each start date (the missing fraction is the mean of
  the shard values weighted by their number of start dates);
- Lagrangian deviation: the mean over the start dates of the deviation of each start date;
- Class IV: pooled again from the match-ups of all the start dates;
- viewer data: built by ``build_viewer_data`` from the merged match-ups and score tables, maps included; the eddy
  census files of the shards are copied with their start index in the year, instead of being detected again.

``eddy-census`` and ``eddy-merge`` do the same for the eddy census alone, to add it to viewer data built without
it: each shard detects the eddies of its start dates, and the merge copies them with their start index in the year
into ``insights/<challenger>/<region>/`` of the viewer directory. Nothing else of the viewer data is written.
"""

import argparse
import contextlib
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy
import pandas

from oceanbench import metrics
from oceanbench.core import evaluate, lagrangian_trajectory
from oceanbench.core.classIV import _rmsd_table_from_parts
from oceanbench.core.classIV_matchups import Class4Matchups
from oceanbench.core.classIV_support import format_class4_results
from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.lagrangian_support import LAGRANGIAN_ROW_LABEL, mean_weekly_lagrangian_deviations
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.metrics import CLASS4_VARIABLES
from oceanbench.core.scores import (
    CLASS4_METRIC_FUNCTION_NAME,
    GRIDDED_METRIC_FUNCTION_NAMES,
    METRIC_FUNCTION_NAMES,
    SCORES_SCHEMA_VERSION,
    score_rows,
    scores_file_name,
    write_scores_file,
)
from oceanbench.core.schema_validation import validate_against_schema
from oceanbench.publish import viewer_artifacts
from oceanbench.publish.viewer_artifacts import EDDY_CENSUS_FILENAME

START_DATES_FILE = "start-dates.json"
START_TABLES_FILE = "gridded-start-tables.pickle"
LAGRANGIAN_FILE = "lagrangian-deviations.json"
MATCHUPS_DIRECTORY = "class4-matchups"
MATCHUPS_INDEX_FILE = "index.json"
LAGRANGIAN_METRIC_FUNCTION_NAMES = [
    name for name in METRIC_FUNCTION_NAMES if name.startswith("deviation_of_lagrangian")
]


def _start_dates(dataset) -> list[str]:
    return [pandas.Timestamp(value).date().isoformat() for value in dataset[Dimension.FIRST_DAY_DATETIME.key()].values]


def _write_json(document, path: Path) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- shard


def _write_matchups(matchups: Class4Matchups | None, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    import pyarrow.feather

    parts = []
    if matchups is not None:
        for variable in matchups.variables():
            for first_day in matchups.first_days():
                for lead_day in matchups.lead_days(variable):
                    part = matchups.part(variable, lead_day, first_day)
                    if not len(part):
                        continue
                    file_name = f"{variable}.{first_day.date().isoformat()}.{lead_day}.arrow"
                    pyarrow.feather.write_feather(
                        part.reset_index(drop=True), directory / file_name, compression="zstd"
                    )
                    parts.append(
                        {
                            "variable": variable,
                            "first_day": first_day.date().isoformat(),
                            "lead_day": int(lead_day),
                            "file": file_name,
                        }
                    )
    _write_json({"has_matchups": matchups is not None, "parts": parts}, directory / MATCHUPS_INDEX_FILE)


def _mark_steps(markers_file: str) -> None:
    """Append "epoch begin|end step tasks computes" lines to a file, so the job's memory samples can be matched to the
    metric running; tasks is the sum of the dask graph sizes computed in the step."""
    import time

    from dask.callbacks import Callback

    counts = {"tasks": 0, "computes": 0}

    def write(event: str, step: str) -> None:
        line = f"{time.time():.1f} {event} {step} {counts['tasks']} {counts['computes']}"
        with open(markers_file, "a") as file:
            file.write(line + "\n")
        print(f"[step marker] {line}", flush=True)

    def count_tasks(dsk) -> None:
        counts["tasks"] += len(dsk)
        counts["computes"] += 1

    Callback(start=count_tasks).register()
    original_timed_step = evaluate._timed_step

    def marked_step(timings, step, function):
        counts.update(tasks=0, computes=0)
        write("begin", step)
        try:
            return original_timed_step(timings, step, function)
        finally:
            write("end", step)

    evaluate._timed_step = marked_step
    original_build_viewer_data = evaluate._build_viewer_data

    def marked_viewer(*positional, **named):
        counts.update(tasks=0, computes=0)
        write("begin", "viewer_data")
        try:
            return original_build_viewer_data(*positional, **named)
        finally:
            write("end", "viewer_data")

    evaluate._build_viewer_data = marked_viewer


def evaluate_shard(arguments: argparse.Namespace) -> None:
    merge_inputs = Path(arguments.merge_inputs_dir)
    merge_inputs.mkdir(parents=True, exist_ok=True)
    if arguments.step_markers:
        _mark_steps(arguments.step_markers)
    lagrangian: dict[str, dict] = {}
    current_metric: list[str] = []

    def lagrangian_metric(name: str):
        original = getattr(metrics, name)

        def wrapped(*positional, **named):
            current_metric.append(name)
            try:
                return original(*positional, **named)
            finally:
                current_metric.pop()

        return wrapped

    for name in LAGRANGIAN_METRIC_FUNCTION_NAMES:
        setattr(metrics, name, lagrangian_metric(name))

    original_points = lagrangian_trajectory._get_random_ocean_points_from_file

    def recorded_points(*positional, **named):
        latitudes, longitudes = original_points(*positional, **named)
        digest = hashlib.sha256(numpy.ascontiguousarray(latitudes).tobytes())
        digest.update(numpy.ascontiguousarray(longitudes).tobytes())
        lagrangian.setdefault(current_metric[-1], {})["particles_sha256"] = digest.hexdigest()
        return latitudes, longitudes

    def recorded_mean(weekly_deviations):
        weekly_deviations = list(weekly_deviations)
        record = lagrangian.setdefault(current_metric[-1], {})
        record["by_start"] = [numpy.asarray(deviations).tolist() for deviations in weekly_deviations]
        record["dtypes"] = [str(numpy.asarray(deviations).dtype) for deviations in weekly_deviations]
        return mean_weekly_lagrangian_deviations(weekly_deviations)

    lagrangian_trajectory._get_random_ocean_points_from_file = recorded_points
    lagrangian_trajectory.mean_weekly_lagrangian_deviations = recorded_mean

    # The maps need the value range of the whole year: the merge builds them.
    viewer_artifacts.build_column_store = lambda *positional, **named: None
    viewer_artifacts.build_pyramid = lambda *positional, **named: None

    original_build_viewer_data = evaluate._build_viewer_data

    def build_viewer_data_and_keep_merge_inputs(
        directory, challenger_name, challenger_dataset, region, matchups, document, start_tables=None
    ):
        _write_json(_start_dates(challenger_dataset), merge_inputs / START_DATES_FILE)
        with open(merge_inputs / START_TABLES_FILE, "wb") as file:
            pickle.dump(start_tables or {}, file)
        _write_matchups(matchups, merge_inputs / MATCHUPS_DIRECTORY)
        _write_json(lagrangian, merge_inputs / LAGRANGIAN_FILE)
        original_build_viewer_data(
            directory, challenger_name, challenger_dataset, region, matchups, document, start_tables
        )

    evaluate._build_viewer_data = build_viewer_data_and_keep_merge_inputs
    evaluate.evaluate_challenger(
        arguments.challenger,
        region=arguments.region or None,
        viewer_artifacts_directory=arguments.viewer_artifacts_dir,
    )


# ---------------------------------------------------------------- merge


class _Shard:
    def __init__(self, directory: Path, challenger_name: str, region_id: str) -> None:
        self.directory = directory
        self.merge_inputs = directory / "merge-inputs"
        self.start_dates = _read_json(self.merge_inputs / START_DATES_FILE)
        self.document = _read_json(directory / f"{challenger_name}.{region_id}.scores.json")
        with open(self.merge_inputs / START_TABLES_FILE, "rb") as file:
            self.start_tables = pickle.load(file)
        self.lagrangian = _read_json(self.merge_inputs / LAGRANGIAN_FILE)
        self.matchups_index = _read_json(self.merge_inputs / MATCHUPS_DIRECTORY / MATCHUPS_INDEX_FILE)
        self.insights = directory / "viewer" / "insights" / challenger_name / region_id


def _check_start_dates(shards: list[_Shard], year_start_dates: list[str]) -> None:
    shard_start_dates = [start_date for shard in shards for start_date in shard.start_dates]
    if shard_start_dates != year_start_dates:
        raise ValueError(
            f"The shards hold the start dates {shard_start_dates}, in this order, "
            f"but the challenger has {year_start_dates}: give the shards in start date order, without gap or overlap"
        )


def _merged_start_tables(shards: list[_Shard]) -> dict[str, list[pandas.DataFrame]]:
    names = [name for name in GRIDDED_METRIC_FUNCTION_NAMES if any(name in shard.start_tables for shard in shards)]
    for name in names:
        if not all(name in shard.start_tables for shard in shards):
            raise ValueError(f"{name} has start tables in some shards only")
    return {name: [table for shard in shards for table in shard.start_tables[name]] for name in names}


def _gridded_rows(metric_function_name: str, start_tables: list[pandas.DataFrame]) -> list[dict]:
    rows_by_start = [score_rows(metric_function_name, table) for table in start_tables]
    merged_rows = []
    for position, row in enumerate(rows_by_start[0]):
        start_rows = [rows[position] for rows in rows_by_start]
        if any(
            (start_row["standard_name"], start_row["depth"], start_row["lead_day"])
            != (row["standard_name"], row["depth"], row["lead_day"])
            for start_row in start_rows
        ):
            raise ValueError(f"{metric_function_name}: the start tables do not have the same rows")
        values = numpy.array([numpy.nan if r["value"] is None else r["value"] for r in start_rows], dtype=float)
        merged_row = dict(row)
        merged_row["value"] = None if not numpy.isfinite(values).any() else float(numpy.nanmean(values))
        if "missing_fraction" in row:
            fractions = numpy.array(
                [numpy.nan if r["missing_fraction"] is None else r["missing_fraction"] for r in start_rows], dtype=float
            )
            merged_row["missing_fraction"] = None if numpy.isnan(fractions).all() else float(numpy.nanmean(fractions))
        merged_rows.append(merged_row)
    return merged_rows


def _lagrangian_rows(metric_function_name: str, shards: list[_Shard], lead_days_count: int) -> list[dict]:
    records = [shard.lagrangian.get(metric_function_name) for shard in shards]
    if all(record is None for record in records):
        return []
    if any(record is None for record in records):
        raise ValueError(f"{metric_function_name} was computed in some shards only")
    digests = {record.get("particles_sha256") for record in records}
    if len(digests) > 1:
        print(f"WARNING: {metric_function_name}: the shards seeded different particles", file=sys.stderr)
    by_start = [
        numpy.array(deviations, dtype=dtype)
        for record in records
        for deviations, dtype in zip(record["by_start"], record.get("dtypes") or ["float64"] * len(record["by_start"]))
    ]
    deviations = mean_weekly_lagrangian_deviations(by_start)
    start = lagrangian_trajectory.LEAD_DAY_START
    table = pandas.DataFrame({LAGRANGIAN_ROW_LABEL: deviations[start - 1 : lead_days_count]})
    table.index = lead_day_labels(start, lead_days_count)
    return score_rows(metric_function_name, table.T)


def _merged_matchups(shards: list[_Shard]) -> Class4Matchups | None:
    if not any(shard.matchups_index["has_matchups"] for shard in shards):
        return None
    import pyarrow.feather

    parts: dict[tuple[str, str], list[tuple[int, Path]]] = {}
    for shard in shards:
        for part in shard.matchups_index["parts"]:
            parts.setdefault((part["variable"], part["first_day"]), []).append(
                (part["lead_day"], shard.merge_inputs / MATCHUPS_DIRECTORY / part["file"])
            )
    variable_order = [variable.key() for variable in CLASS4_VARIABLES]
    matchups = Class4Matchups.on_disk(variable_order)
    for variable in variable_order:
        for first_day in sorted({first_day for part_variable, first_day in parts if part_variable == variable}):
            frames = [pyarrow.feather.read_feather(path) for _, path in sorted(parts[(variable, first_day)])]
            matchups.add(variable, pandas.concat(frames, ignore_index=True))
    return matchups


def _class4_rows(matchups: Class4Matchups | None, shards: list[_Shard], lead_days_count: int) -> list[dict]:
    if matchups is None or not len(matchups):
        return [row for row in shards[0].document["rows"] if row["reference"] == "observations"]
    tables = [_rmsd_table_from_parts(matchups, variable) for variable in matchups.variables()]
    tables = [table for table in tables if not table.empty]
    if not tables:
        return []
    return score_rows(
        CLASS4_METRIC_FUNCTION_NAME, format_class4_results(pandas.concat(tables, ignore_index=True), lead_days_count)
    )


def _merged_timings(shards: list[_Shard]) -> dict[str, float]:
    timings: dict[str, float] = {}
    for shard in shards:
        for step, duration in shard.document.get("timings", {}).items():
            timings[step] = timings.get(step, 0.0) + duration
    return timings


def _copy_eddy_census(shards: list[_Shard], year_start_dates: list[str]):
    """``write_region_eddy_census`` for the merge: the files of the shards with their start index in the year."""

    def write_region_eddy_census(insights_directory, dataset_slug, challenger_dataset, region):
        indexes = [shard.insights / EDDY_CENSUS_FILENAME for shard in shards]
        present = [index.exists() for index in indexes]
        if not any(present):
            return None
        if not all(present):
            raise ValueError("The eddy census is in some shards only")
        year_positions = {start_date: position for position, start_date in enumerate(year_start_dates)}
        metadata = None
        start_entries = []
        for shard, index_path in zip(shards, indexes):
            index = _read_json(index_path)
            shard_metadata = {key: value for key, value in index.items() if key not in ("starts", "leads")}
            if metadata is None:
                metadata = shard_metadata
            elif shard_metadata != metadata:
                raise ValueError(f"The eddy census of {shard.directory} has other parameters")
            for entry in index["starts"]:
                start_index = year_positions[entry["start_date"]]
                for lead in entry["leads"]:
                    payload = _read_json(shard.insights / lead["file"])
                    payload["start_index"] = start_index
                    (insights_directory / lead["file"]).write_text(
                        json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
                    )
                start_entries.append({**entry, "start_index": start_index})
        index = {**metadata, "starts": start_entries, "leads": start_entries[0]["leads"] if start_entries else []}
        (insights_directory / EDDY_CENSUS_FILENAME).write_text(
            json.dumps(index, sort_keys=True, indent=2), encoding="utf-8"
        )
        return EDDY_CENSUS_FILENAME

    return write_region_eddy_census


def eddy_census(arguments: argparse.Namespace) -> None:
    region = evaluate.resolve_evaluation_region(arguments.region or None)
    challenger_name = PurePosixPath(arguments.challenger).stem
    challenger_dataset = evaluate.open_challenger_dataset(evaluate.read_challenger_python_code(arguments.challenger))
    insights_directory = Path(arguments.viewer_artifacts_dir) / "insights" / challenger_name / region.id
    insights_directory.mkdir(parents=True, exist_ok=True)
    if viewer_artifacts.write_region_eddy_census(insights_directory, challenger_name, challenger_dataset, region) is None:
        raise SystemExit("No eddy census possible: the grid is too coarse or py-eddy-tracker is not installed")
    print(f"Eddy census of {len(_start_dates(challenger_dataset))} start dates written to {insights_directory}")


class _EddyShard:
    def __init__(self, directory: Path, challenger_name: str, region_id: str) -> None:
        self.directory = directory
        self.insights = directory / "viewer" / "insights" / challenger_name / region_id
        index = _read_json(self.insights / EDDY_CENSUS_FILENAME)
        self.start_dates = [entry["start_date"] for entry in index["starts"]]


def eddy_merge(arguments: argparse.Namespace) -> None:
    region = evaluate.resolve_evaluation_region(arguments.region or None)
    challenger_name = PurePosixPath(arguments.challenger).stem
    challenger_dataset = evaluate.open_challenger_dataset(evaluate.read_challenger_python_code(arguments.challenger))
    year_start_dates = _start_dates(challenger_dataset)
    shards = [_EddyShard(Path(directory), challenger_name, region.id) for directory in arguments.shards]
    _check_start_dates(shards, year_start_dates)
    insights_directory = Path(arguments.viewer_artifacts_dir) / "insights" / challenger_name / region.id
    insights_directory.mkdir(parents=True, exist_ok=True)
    _copy_eddy_census(shards, year_start_dates)(insights_directory, challenger_name, challenger_dataset, region)
    print(f"Eddy census of {len(year_start_dates)} start dates from {len(shards)} shards written to {insights_directory}")


def merge(arguments: argparse.Namespace) -> None:
    region = evaluate.resolve_evaluation_region(arguments.region or None)
    challenger_name = PurePosixPath(arguments.challenger).stem
    challenger_python_code = evaluate.read_challenger_python_code(arguments.challenger)
    challenger_dataset = evaluate.open_challenger_dataset(challenger_python_code)
    year_start_dates = _start_dates(challenger_dataset)
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    shards = [_Shard(Path(directory), challenger_name, region.id) for directory in arguments.shards]
    _check_start_dates(shards, year_start_dates)

    start_tables = _merged_start_tables(shards)
    matchups = _merged_matchups(shards)
    rows = []
    for name in METRIC_FUNCTION_NAMES:
        if name in start_tables:
            rows += _gridded_rows(name, start_tables[name])
        elif name == CLASS4_METRIC_FUNCTION_NAME:
            rows += _class4_rows(matchups, shards, lead_days_count)
        elif name in LAGRANGIAN_METRIC_FUNCTION_NAMES:
            rows += _lagrangian_rows(name, shards, lead_days_count)
        else:
            rows += [
                row for row in shards[0].document["rows"] if f"{row['metric']}_compared_to_{row['reference']}" == name
            ]
    document = {
        **shards[0].document,
        "challenger_code": challenger_python_code,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "timings": _merged_timings(shards),
        "rows": rows,
    }
    validate_against_schema(document, f"scores-v{SCORES_SCHEMA_VERSION}")
    output_file = Path(arguments.output_dir) / scores_file_name(challenger_name, region)
    write_scores_file(document, str(output_file))
    print(f"Scores of {len(year_start_dates)} start dates from {len(shards)} shards written to {output_file}")

    if arguments.viewer_artifacts_dir:
        from oceanbench.core.challenger_datasets import field_chunked_challenger_open
        from oceanbench.publish.viewer_data import build_viewer_data

        with field_chunked_challenger_open():
            map_dataset = evaluate.open_challenger_dataset(challenger_python_code)
        viewer_artifacts.write_region_eddy_census = _copy_eddy_census(shards, year_start_dates)
        build_viewer_data(
            arguments.viewer_artifacts_dir,
            challenger_name,
            challenger_dataset,
            region,
            matchups,
            document["rows"],
            start_tables or None,
            map_dataset,
        )
        print(f"Viewer data of {challenger_name} written to {arguments.viewer_artifacts_dir}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shards")
    commands = parser.add_subparsers(dest="command", required=True)
    shard = commands.add_parser("evaluate-shard")
    shard.add_argument("challenger")
    shard.add_argument("--viewer-artifacts-dir", required=True)
    shard.add_argument("--merge-inputs-dir", required=True)
    shard.add_argument("--region", default="")
    shard.add_argument("--step-markers", default="", help="file to append the begin and end time of each step to")
    shard.set_defaults(function=evaluate_shard)
    merge_command = commands.add_parser("merge")
    merge_command.add_argument("challenger")
    merge_command.add_argument("shards", nargs="+", help="shard directories, in start date order")
    merge_command.add_argument("--output-dir", required=True)
    merge_command.add_argument("--viewer-artifacts-dir", default="")
    merge_command.add_argument("--region", default="")
    merge_command.set_defaults(function=merge)
    eddy_command = commands.add_parser("eddy-census")
    eddy_command.add_argument("challenger")
    eddy_command.add_argument("--viewer-artifacts-dir", required=True)
    eddy_command.add_argument("--region", default="")
    eddy_command.set_defaults(function=eddy_census)
    eddy_merge_command = commands.add_parser("eddy-merge")
    eddy_merge_command.add_argument("challenger")
    eddy_merge_command.add_argument("shards", nargs="+", help="eddy census shard directories, in start date order")
    eddy_merge_command.add_argument("--viewer-artifacts-dir", required=True)
    eddy_merge_command.add_argument("--region", default="")
    eddy_merge_command.set_defaults(function=eddy_merge)
    return parser


if __name__ == "__main__":
    parsed_arguments = _parser().parse_args()
    with contextlib.suppress(BrokenPipeError):
        parsed_arguments.function(parsed_arguments)
