# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Compare rung L_WS against the default-policy rung L3 on identical matchup rows."""
import glob, os
import numpy, pandas

R = "/scratch/jseillade/obs-rebuild/rescore"
OUT = os.path.join(R, "ws-rung")
SHORT = {"eastward_sea_water_velocity": "uo", "northward_sea_water_velocity": "vo"}

frames = []
for pattern in (os.path.join(R, "results/glonet/L3_*.csv"), os.path.join(R, "results/glonet-ws/*.csv")):
    for f in sorted(glob.glob(pattern)):
        frames.append(pandas.read_csv(f))
frame = pandas.concat(frames, ignore_index=True)
frame = frame[frame["variable"].isin(SHORT)]
frame["variable"] = frame["variable"].map(SHORT)
frame["lead"] = frame["lead_day"].astype(int) + 1
frame = frame[frame["count"] > 0]

pooled = (
    frame.groupby(["rung", "region", "variable", "lead"], as_index=False)
    .agg(sumsq=("sumsq", "sum"), count=("count", "sum"))
)
pooled["rmsd"] = numpy.sqrt(pooled["sumsq"] / pooled["count"])

# combined current-vector rmsd: sqrt((sumsq_uo + sumsq_vo) / (count_uo + count_vo))
combined = (
    pooled.groupby(["rung", "region", "lead"], as_index=False)
    .agg(sumsq=("sumsq", "sum"), count=("count", "sum"))
)
combined["variable"] = "uv"
combined["rmsd"] = numpy.sqrt(combined["sumsq"] / combined["count"])
pooled = pandas.concat([pooled, combined], ignore_index=True)
pooled.to_csv(os.path.join(OUT, "rmsd_pooled_ws.csv"), index=False)

index = pooled.set_index(["rung", "region", "variable", "lead"])
rows = []
for (region, variable, lead), _ in pooled.groupby(["region", "variable", "lead"]):
    try:
        base = index.loc[("L3", region, variable, lead)]
    except KeyError:
        continue
    entry = {"region": region, "variable": variable, "lead": lead,
             "L3_rmsd": float(base["rmsd"]), "L3_count": int(base["count"])}
    for rung in ("LWS", "LWSp"):
        try:
            row = index.loc[(rung, region, variable, lead)]
        except KeyError:
            continue
        entry[f"{rung}_rmsd"] = float(row["rmsd"])
        entry[f"{rung}_count"] = int(row["count"])
        entry[f"{rung}_delta"] = float(row["rmsd"]) - float(base["rmsd"])
        entry[f"{rung}_percent"] = 100.0 * entry[f"{rung}_delta"] / float(base["rmsd"])
    rows.append(entry)
table = pandas.DataFrame(rows).sort_values(["region", "variable", "lead"])
table.to_csv(os.path.join(OUT, "comparison.csv"), index=False)

lines = ["# Wind slippage rung, challenger glonet", "",
         "L3 = default v2 policy (currents on the EWCT_FILTR basis, no WS subtraction).",
         "LWS = same rows, currents minus the wind slippage columns where finite.",
         "LWSp = sign-check variant, currents plus WS. Diagnostic only.",
         "RMSD in m/s, pooled over depth bins. Counts are matchup rows at that lead.", ""]
for region in ["global", "ibi"]:
    lines += [f"## region {region}", ""]
    header = ["variable", "lead", "n", "L3 rmsd", "LWS rmsd", "LWS delta", "LWS %", "LWSp rmsd", "LWSp %"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    sub = table[(table["region"] == region) & (table["lead"].isin([1, 5, 9]))]
    for _, r in sub.iterrows():
        lines.append("| " + " | ".join([
            r["variable"], str(int(r["lead"])), str(int(r["L3_count"])),
            f"{r[chr(76)+chr(51)+chr(95)+chr(114)+chr(109)+chr(115)+chr(100)]:.5f}",
            f"{r[chr(76)+chr(87)+chr(83)+chr(95)+chr(114)+chr(109)+chr(115)+chr(100)]:.5f}",
            f"{r[chr(76)+chr(87)+chr(83)+chr(95)+chr(100)+chr(101)+chr(108)+chr(116)+chr(97)]:+.5f}",
            f"{r[chr(76)+chr(87)+chr(83)+chr(95)+chr(112)+chr(101)+chr(114)+chr(99)+chr(101)+chr(110)+chr(116)]:+.2f}%",
            f"{r[chr(76)+chr(87)+chr(83)+chr(112)+chr(95)+chr(114)+chr(109)+chr(115)+chr(100)]:.5f}",
            f"{r[chr(76)+chr(87)+chr(83)+chr(112)+chr(95)+chr(112)+chr(101)+chr(114)+chr(99)+chr(101)+chr(110)+chr(116)]:+.2f}%",
        ]) + " |")
    lines.append("")
with open(os.path.join(OUT, "RESULTS.md"), "w") as handle:
    handle.write("\n".join(lines) + "\n")
print("\n".join(lines))
count_mismatch = table[(table["L3_count"] != table["LWS_count"])]
print("count mismatches L3 vs LWS:", len(count_mismatch))
if len(count_mismatch):
    print(count_mismatch.to_string(index=False))
