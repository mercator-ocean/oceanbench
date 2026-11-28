<!--
SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# OceanBench evaluation as an EDITO process

`run-evaluation.sh` runs `oceanbench evaluate` (scores and, optionally, the viewer data) in the EDITO "generic process" job, then uploads the results to S3.
It installs OceanBench from the cloned branch itself, so the job evaluates exactly the code of that branch.

## Form values

In the EDITO datalab, launch the "Generic process" service (catalog "Processes") and fill in:

| Field | Value |
|---|---|
| Source | My code is based on a Python or Shell script |
| Git repository | `https://github.com/mercator-ocean/oceanbench.git` |
| Git branch | the branch to evaluate, for example `main` |
| Entrypoint path | `edito/process/run-evaluation.sh` |
| Environment path | `edito/process/environment.yaml` |
| Target | My process should execute on EDITO Cloud, no GPU |
| CPU request / limit | `7200m` / `7200m` |
| Memory request / limit | `28Gi` / `28Gi` |
| S3 | enabled (the uploads use the injected `AWS_*` variables) |
| Copernicus Marine | enabled (the references and observations need an account) |

The paths are relative to the root of the repository: the job runs the entry script from there.

## Process inputs

Each input is an environment variable of the job.
Only `CHALLENGER` is required; leave the others out to use their default.

| Name | Example | Meaning |
|---|---|---|
| `CHALLENGER` | `glonet` | A file name of `challenger_datasets` without `.py`, a path in the repository (`assets/glonet_sample.py`) or an `https` URL of a challenger file |
| `START_DATES` | `2024-01-03,2024-01-10` | Evaluate only these start dates (comma separated) |
| `START_COUNT` | `4` | Evaluate only the first N start dates (not with `START_DATES`) |
| `RUN_NAME` | `glonet` | Name of the scores file and of the viewer dataset, by default the challenger file name |
| `REGION` | `ibi` | Official region, `global` by default |
| `OUTPUT_BUCKET` | `oceanbench-bucket` | Bucket to upload to. Without it nothing touches S3 and the scores are printed to the log (see Logs) |
| `UPLOAD` | `true` | `false` ignores `OUTPUT_BUCKET`: no S3 check, no upload |
| `PRINT_SCORES` | `auto` | Print the scores to the log: `auto` (only without upload), `true` or `false` |
| `OUTPUT_PREFIX` | `dev/benchmark/glonet-2024` | Prefix to upload under, required with `OUTPUT_BUCKET` |
| `OVERWRITE` | `false` | The job stops before computing anything if the prefix already holds objects, unless this is `true` |
| `VIEWER` | `true` | Also build the viewer data (`--viewer-artifacts-dir`) |
| `VIEWER_ONLY` | `false` | Rebuild only the viewer data from an existing scores file |
| `SCORES_FILE` | `s3://oceanbench-bucket/dev/benchmark/glonet-2024/glonet.global.scores.json` | With `VIEWER_ONLY=true`: the scores file to read (`s3://`, `https://` or a path). By default the scores file under `OUTPUT_PREFIX` |
| `DASK_NUM_WORKERS` | `16` | Dask threads. By default twice the CPU limit, at most 32 (the threads mostly wait on remote reads) |
| `MEMORY_ALERT_GB` | `20` | Every 5 s memory sample whose anonymous memory is above this is logged as a `[memory high]` line (the minute `[memory]` line also gives the maximum of the 5 s samples since the previous line) |
| `MODE` | `shard` | `evaluate` (default), `shard`, `merge`, `eddy` or `eddy-merge`, see "Year in several jobs" |
| `SHARD_PREFIXES` | `dev/x/glonet/shard-1,dev/x/glonet/shard-2` | With `MODE=merge` or `MODE=eddy-merge`: the `OUTPUT_PREFIX` of each shard job in `OUTPUT_BUCKET`, in start date order |
| `OCEANBENCH_PY_EDDY_TRACKER_SPEC` | `git+https://github.com/Murkyy/py-eddy-tracker@3753962e9553f0fb0fd47219afadd1a019bff30a` | A pip installable py-eddy-tracker, for the eddy census of grids finer than 0.3 degree. The example is a fork of the upstream master for numpy 2 and matplotlib 3.10 (Polygon3 builds from source with the gcc of the environment). The job fails if it does not import; without it the eddy census is skipped, with a log line saying so |

The other `OCEANBENCH_*` variables of OceanBench (for example `OCEANBENCH_EDDY_CENSUS_MAX_WORKERS`, `OCEANBENCH_STAGE_MAX_WORKERS`, `OCEANBENCH_REMOTE_RETRIES`) can be given as inputs too.
By default the number of eddy census workers is set by OceanBench from the memory left to the job.

## Outputs

Under `s3://OUTPUT_BUCKET/OUTPUT_PREFIX/`:

- `<name>.<region>.scores.json`, the scores file;
- `viewer/`, the viewer data, uploaded data first and the index and manifest files last, so that a reader never sees an index pointing to missing files.

The S3 variables EDITO injects are those of your EDITO storage: to write into another bucket, its credentials must be the ones in the `AWS_*` variables of the job.

## Year in several jobs

A year does not fit in one job's memory and time for most models, so it can be split by start dates.
Each `MODE=shard` job evaluates the `START_DATES` it is given, with the viewer data but without the maps, and also uploads `merge-inputs/` (the gridded scores, missing fractions and Lagrangian deviations of each start date and the Class IV match-ups).
Then one `MODE=merge` job, with the same `CHALLENGER` and no `START_DATES` (or all the start dates of the shards), reads the shards listed in `SHARD_PREFIXES`, writes the scores file of the year and builds the viewer data of the year, maps included, as one evaluation of all the start dates would (see `shards.py`).

To add the eddy census alone to viewer data built without it, the `MODE=eddy` jobs detect the eddies of their `START_DATES` (only the sea surface height is read, no scores), then one `MODE=eddy-merge` job with `SHARD_PREFIXES` writes the eddy files of the year with their start index under `viewer/insights/<challenger>/<region>/`. Nothing else of the viewer data is written: `insights.json` is not, its `eddies` entry is set when the files are added to the published viewer data. Both modes need `VIEWER=true` and py-eddy-tracker (`OCEANBENCH_PY_EDDY_TRACKER_SPEC`).
The shards must hold every start date of the challenger once, in order. For `MODE=merge`, every shard but the last must hold an even number of start dates (for 52, for example 14, 14, 12 and 12): the geostrophic currents are computed by pairs of start dates, and the merged scores are then bit for bit the ones of one evaluation.

## Logs

The log gives, for each stage (install, evaluate, upload), its start and end time, its duration, the peak memory of the job in the stage and overall, and the free disk.
It starts with the CPU and memory limits seen by the job.
The job exits with a non-zero status when any stage fails.

A successful job ends with a block between `BEGIN JOB SUMMARY` and `END JOB SUMMARY`: the wall time of each stage, the total, and the peak memory (anonymous, with page cache, and the cgroup peak).
Without upload, it then prints the scores file gzipped and base64-encoded between `BEGIN SCORES JSON GZIP BASE64` and `END SCORES JSON GZIP BASE64` (about 16 kB for two GLONET starts), or a mean per metric and reference between `BEGIN SCORES MEAN TABLE` and `END SCORES MEAN TABLE` when the encoded file is over 2 MB.
To get the file back from a saved log:

```bash
sed -n '/^BEGIN SCORES JSON/,/^END SCORES JSON/p' job.log | sed '1d;$d' | base64 -d | gunzip > scores.json
```

## Size

GLONET (1/4 degree), two starts, scores and viewer data without eddies, 7.2 CPUs and 28 GB: 8 minutes in all (environment and install 1 minute, evaluation 6.5 minutes, upload 8 seconds), peak memory 5.7 GB.
One 1/12 degree start with the eddy census (8 workers), 8 CPUs and 32 GB: about 20 minutes, peak under 20 GB.
A full year (52 starts) of a 1/12 degree model is about 18 hours in one job.
