# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Old basis versus new basis comparison for the 2024-v2.1.0 promotion rescore.

Reads the per-chunk sum-of-squares CSVs written by score_rung.py under
results/<challenger>/ and emits, per challenger, per stream, per region:
the OLD (legacy observations2024) RMSD, the NEW (2024-v2.1.0) RMSD, the
absolute delta and the percent change, both pooled over every lead and depth
bin and per headline lead.

Both sides come from the same scoring code path and the same chunking, so the
delta is a basis change only.
"""
import glob
import os
import sys

import numpy
import pandas

BASE = "/scratch/jseillade/obs-rebuild/rescore/full-rescore-v21"
RESULTS = os.path.join(BASE, "results")
TABLES = os.path.join(BASE, "tables")
HEADLINE_LEADS = (1, 5, 9)
SHORT = {
    "sea_water_potential_temperature": "temperature",
    "sea_water_salinity": "salinity",
    "sea_surface_height_above_geoid": "sla",
    "eastward_sea_water_velocity": "uo",
    "northward_sea_water_velocity": "vo",
}
STREAM_ORDER = ("temperature", "salinity", "sla", "uo", "vo")


def load(only):
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*", "*.csv"))):
        challenger = os.path.basename(os.path.dirname(path))
        if only and challenger not in only:
            continue
        frame = pandas.read_csv(path)
        if frame.empty:
            continue
        frame["challenger"] = challenger
        rows.append(frame)
    if not rows:
        sys.exit(f"no result csv under {RESULTS} for {sorted(only)}")
    frame = pandas.concat(rows, ignore_index=True)
    frame = frame[frame["count"] > 0]
    frame["variable"] = frame["variable"].map(SHORT).fillna(frame["variable"])
    frame["lead"] = frame["lead_day"].astype(int) + 1
    return frame


def rmsd_table(frame, keys):
    table = frame.groupby(keys, as_index=False).agg(sumsq=("sumsq", "sum"), count=("count", "sum"))
    table["rmsd"] = numpy.sqrt(table["sumsq"] / table["count"])
    return table.drop(columns=["sumsq"])


def widen(table, keys):
    wide = table.pivot_table(index=keys, columns="rung", values=["rmsd", "count"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for side in ("OLD", "NEW"):
        for column in (f"rmsd_{side}", f"count_{side}"):
            if column not in wide:
                wide[column] = numpy.nan
    wide["delta"] = wide["rmsd_NEW"] - wide["rmsd_OLD"]
    wide["percent"] = 100.0 * wide["delta"] / wide["rmsd_OLD"]
    return wide


def main():
    only = set(sys.argv[1:])
    os.makedirs(TABLES, exist_ok=True)
    frame = load(only)

    coverage = (
        frame.groupby(["challenger", "rung", "region"])["chunk"]
        .nunique().rename("chunks_present").reset_index()
    )
    coverage.to_csv(os.path.join(TABLES, "coverage.csv"), index=False)

    keys = ["challenger", "region", "variable"]
    pooled = widen(rmsd_table(frame, keys + ["rung"]), keys)
    pooled.sort_values(keys).to_csv(os.path.join(TABLES, "compare_pooled.csv"), index=False)

    per_lead = widen(rmsd_table(frame, keys + ["lead", "rung"]), keys + ["lead"])
    per_lead.sort_values(keys + ["lead"]).to_csv(os.path.join(TABLES, "compare_per_lead.csv"),
                                                 index=False)

    per_bin = rmsd_table(frame, keys + ["depth_bin", "lead", "rung"])
    per_bin.sort_values(keys + ["depth_bin", "lead"]).to_csv(
        os.path.join(TABLES, "rmsd_per_depth_bin.csv"), index=False)

    lines = ["# Old basis versus new basis, class-4 RMSD", ""]
    lines.append("OLD = legacy observations2024. NEW = observations2024-v2 at 2024-v2.1.0")
    lines.append("(FILTR minus wind slippage, current_test 11 and 211 dropped).")
    lines.append("RMSD pooled over depth bins and leads unless a lead is named.")
    lines.append("Percent is the change from OLD to NEW; negative means the new basis scores better.")
    lines.append("")
    lines.append("Chunks present per challenger, rung and region:")
    lines.append("")
    lines.append("| " + " | ".join(coverage.columns) + " |")
    lines.append("| " + " | ".join("---" for _ in coverage.columns) + " |")
    for _, row in coverage.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")

    for region in sorted(pooled["region"].unique()):
        lines += ["", f"## region {region}", "", "### pooled over all leads", ""]
        header = ["challenger", "stream", "OLD rmsd", "NEW rmsd", "delta", "%",
                  "n OLD", "n NEW"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        subset = pooled[pooled["region"] == region]
        for challenger in sorted(subset["challenger"].unique()):
            for stream in STREAM_ORDER:
                row = subset[(subset["challenger"] == challenger) & (subset["variable"] == stream)]
                if row.empty:
                    continue
                r = row.iloc[0]
                lines.append("| " + " | ".join([
                    challenger, stream,
                    "" if numpy.isnan(r["rmsd_OLD"]) else f"{r['rmsd_OLD']:.5f}",
                    "" if numpy.isnan(r["rmsd_NEW"]) else f"{r['rmsd_NEW']:.5f}",
                    "" if numpy.isnan(r["delta"]) else f"{r['delta']:+.5f}",
                    "" if numpy.isnan(r["percent"]) else f"{r['percent']:+.2f}%",
                    "" if numpy.isnan(r["count_OLD"]) else f"{int(r['count_OLD'])}",
                    "" if numpy.isnan(r["count_NEW"]) else f"{int(r['count_NEW'])}",
                ]) + " |")

        lines += ["", "### per lead", ""]
        header = ["challenger", "stream"]
        for lead in HEADLINE_LEADS:
            header += [f"L{lead} OLD", f"L{lead} NEW", f"L{lead} %"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        lead_subset = per_lead[per_lead["region"] == region]
        for challenger in sorted(lead_subset["challenger"].unique()):
            for stream in STREAM_ORDER:
                cells = [challenger, stream]
                found = False
                for lead in HEADLINE_LEADS:
                    row = lead_subset[(lead_subset["challenger"] == challenger)
                                      & (lead_subset["variable"] == stream)
                                      & (lead_subset["lead"] == lead)]
                    if row.empty:
                        cells += ["", "", ""]
                        continue
                    found = True
                    r = row.iloc[0]
                    cells += [
                        "" if numpy.isnan(r["rmsd_OLD"]) else f"{r['rmsd_OLD']:.5f}",
                        "" if numpy.isnan(r["rmsd_NEW"]) else f"{r['rmsd_NEW']:.5f}",
                        "" if numpy.isnan(r["percent"]) else f"{r['percent']:+.2f}%",
                    ]
                if found:
                    lines.append("| " + " | ".join(cells) + " |")

    with open(os.path.join(TABLES, "RESULTS.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {TABLES}/RESULTS.md, compare_pooled.csv, compare_per_lead.csv, "
          f"rmsd_per_depth_bin.csv, coverage.csv")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
