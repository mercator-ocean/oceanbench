# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Combine the per-chunk sum-of-squares CSVs into the ladder tables."""
import glob, os, sys
import numpy, pandas

CHALLENGER = sys.argv[1] if len(sys.argv) > 1 else "glonet"
R = "/scratch/jseillade/obs-rebuild/rescore"
RESULTS = os.path.join(R, "results", CHALLENGER)
OUT = os.path.join(R, "tables", CHALLENGER)
os.makedirs(OUT, exist_ok=True)

RUNG_ORDER = ["L0", "L1", "L2", "L2b", "L3", "L1raw"]
# the attribution chain, per variable. L1raw is off-chain and is compared to L1.
CHAIN = {
    "temperature": ["L0", "L1", "L3"],
    "salinity": ["L0", "L1", "L3"],
    "sla": ["L0", "L1", "L3"],
    "uo": ["L0", "L1", "L2", "L2b", "L3"],
    "vo": ["L0", "L1", "L2", "L2b", "L3"],
}
OFF_CHAIN = {"L1raw": "L1"}
SHORT = {
    "sea_water_potential_temperature": "temperature",
    "sea_water_salinity": "salinity",
    "sea_surface_height_above_geoid": "sla",
    "eastward_sea_water_velocity": "uo",
    "northward_sea_water_velocity": "vo",
}

files = sorted(glob.glob(os.path.join(RESULTS, "*.csv")))
if not files:
    sys.exit(f"no result csv under {RESULTS}")
frame = pandas.concat([pandas.read_csv(f) for f in files], ignore_index=True)
frame = frame[frame["count"] > 0]
frame["variable"] = frame["variable"].map(SHORT).fillna(frame["variable"])
frame["lead"] = frame["lead_day"].astype(int) + 1

# how many chunks landed per rung x region, so a partial ladder is never read as final
coverage = (
    frame.groupby(["rung", "region"])["chunk"].nunique().rename("chunks_present").reset_index()
)
coverage.to_csv(os.path.join(OUT, "coverage.csv"), index=False)

per_bin = (
    frame.groupby(["rung", "region", "variable", "depth_bin", "lead"], as_index=False)
    .agg(sumsq=("sumsq", "sum"), count=("count", "sum"))
)
per_bin["rmsd"] = numpy.sqrt(per_bin["sumsq"] / per_bin["count"])
per_bin.sort_values(["region", "variable", "depth_bin", "rung", "lead"]).to_csv(
    os.path.join(OUT, "rmsd_per_depth_bin.csv"), index=False
)

pooled = (
    frame.groupby(["rung", "region", "variable", "lead"], as_index=False)
    .agg(sumsq=("sumsq", "sum"), count=("count", "sum"))
)
pooled["rmsd"] = numpy.sqrt(pooled["sumsq"] / pooled["count"])
pooled = pooled.drop(columns=["sumsq"])
pooled.sort_values(["region", "variable", "rung", "lead"]).to_csv(
    os.path.join(OUT, "rmsd_pooled.csv"), index=False
)

# ladder with deltas against the previous rung that exists for that variable
rows = []
for (region, variable, lead), group in pooled.groupby(["region", "variable", "lead"]):
    available = set(group["rung"])
    by_rung = group.set_index("rung")
    chain = [r for r in CHAIN.get(variable, RUNG_ORDER) if r in available]
    emitted = []
    previous = None
    for rung in chain:
        rmsd = float(by_rung.loc[rung, "rmsd"])
        delta = numpy.nan if previous is None else rmsd - previous[1]
        percent = numpy.nan if previous is None else 100.0 * (rmsd - previous[1]) / previous[1]
        emitted.append(
            {
                "region": region, "variable": variable, "lead": lead, "rung": rung,
                "rmsd": rmsd, "count": int(by_rung.loc[rung, "count"]),
                "previous_rung": None if previous is None else previous[0],
                "delta": delta, "percent": percent,
            }
        )
        previous = (rung, rmsd)
    for rung, reference_rung in OFF_CHAIN.items():
        if rung not in available:
            continue
        rmsd = float(by_rung.loc[rung, "rmsd"])
        has_reference = reference_rung in available
        reference_rmsd = float(by_rung.loc[reference_rung, "rmsd"]) if has_reference else numpy.nan
        emitted.append(
            {
                "region": region, "variable": variable, "lead": lead, "rung": rung,
                "rmsd": rmsd, "count": int(by_rung.loc[rung, "count"]),
                "previous_rung": reference_rung if has_reference else None,
                "delta": rmsd - reference_rmsd,
                "percent": 100.0 * (rmsd - reference_rmsd) / reference_rmsd,
            }
        )
    rows.extend(emitted)
ladder = pandas.DataFrame(rows).sort_values(["region", "variable", "lead", "rung"])
ladder.to_csv(os.path.join(OUT, "ladder.csv"), index=False)

HEADLINE_LEADS = [1, 5, 9]
lines = [f"# Class-4 obs-store ladder, challenger {CHALLENGER}", ""]
lines.append("RMSD pooled over depth bins. delta and % are against the previous rung.")
lines.append("")
lines.append("Chunks present per rung and region:")
lines.append("")
lines.append("| " + " | ".join(coverage.columns) + " |")
lines.append("| " + " | ".join("---" for _ in coverage.columns) + " |")
for _, coverage_row in coverage.iterrows():
    lines.append("| " + " | ".join(str(value) for value in coverage_row) + " |")
for region in sorted(ladder["region"].unique()):
    lines += ["", f"## region {region}", ""]
    for variable in ["temperature", "salinity", "sla", "uo", "vo"]:
        sub = ladder[(ladder["region"] == region) & (ladder["variable"] == variable)]
        if sub.empty:
            continue
        lines += [f"### {variable}", ""]
        header = ["rung"]
        for lead in HEADLINE_LEADS:
            header += [f"L{lead} rmsd", f"L{lead} delta", f"L{lead} %"]
        header += ["n (lead 1)"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for rung in RUNG_ORDER:
            cells = [rung]
            found = False
            for lead in HEADLINE_LEADS:
                row = sub[(sub["rung"] == rung) & (sub["lead"] == lead)]
                if row.empty:
                    cells += ["", "", ""]
                    continue
                found = True
                r = row.iloc[0]
                cells += [
                    f"{r['rmsd']:.5f}",
                    "" if numpy.isnan(r["delta"]) else f"{r['delta']:+.5f}",
                    "" if numpy.isnan(r["percent"]) else f"{r['percent']:+.2f}%",
                ]
            row1 = sub[(sub["rung"] == rung) & (sub["lead"] == 1)]
            cells.append("" if row1.empty else f"{int(row1.iloc[0]['count'])}")
            if found:
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
path = os.path.join(OUT, "ladder.md")
with open(path, "w") as handle:
    handle.write("\n".join(lines) + "\n")
print(f"wrote {OUT}/ladder.csv, rmsd_pooled.csv, rmsd_per_depth_bin.csv, coverage.csv, ladder.md")
print(coverage.to_string(index=False))
