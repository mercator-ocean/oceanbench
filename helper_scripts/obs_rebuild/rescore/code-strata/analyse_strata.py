# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Aggregate the CURRENT_TEST strata and write RESULTS.md."""
import glob
import os

import numpy
import pandas

ROOT = "/scratch/jseillade/obs-rebuild/rescore"
OUT = os.path.join(ROOT, "code-strata")
CODES = [313, 312, 311, 213, 212, 211, 11]
LABEL = {11: "011"}
BANDS = [-80, -60, -40, -20, 0, 20, 40, 60]
RESAMPLES = 2000
RNG = numpy.random.default_rng(20260806)

frame = pandas.concat(
    [pandas.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "results/glonet-strata/*.csv")))],
    ignore_index=True,
)
frame["lead"] = frame["lead_day"].astype(int) + 1
frame = frame[(frame["lead"] >= 1) & (frame["lead"] <= 9)]
frame = frame[frame["count"] > 0]
print("start dates", frame["first_day"].nunique(), "rows", len(frame))


def pooled(selection, keys):
    grouped = selection.groupby(keys, as_index=False).agg(
        sumsq=("sumsq", "sum"), sumres=("sumres", "sum"), count=("count", "sum")
    )
    grouped["rmsd"] = numpy.sqrt(grouped["sumsq"] / grouped["count"])
    grouped["bias"] = grouped["sumres"] / grouped["count"]
    return grouped


def uv_frame(selection):
    """Combine the two components: one row per matchup contributes two residuals."""
    return selection.copy()


def code_label(code):
    return LABEL.get(int(code), str(int(code)))


def rmsd_by_day(selection, code, band=None):
    subset = selection[selection["current_test"] == code]
    if band is not None:
        subset = subset[subset["lat_band"] == band]
    grouped = subset.groupby("first_day", as_index=False).agg(
        sumsq=("sumsq", "sum"), count=("count", "sum")
    )
    return grouped


def bootstrap_difference(selection, code, band=None):
    """Difference in uv RMSD, code minus 313, resampling start dates."""
    left = rmsd_by_day(selection, code, band).set_index("first_day")
    right = rmsd_by_day(selection, 313, band).set_index("first_day")
    days = sorted(set(left.index) & set(right.index))
    if len(days) < 5:
        return None
    left = left.loc[days]
    right = right.loc[days]
    ls, lc = left["sumsq"].to_numpy(), left["count"].to_numpy()
    rs, rc = right["sumsq"].to_numpy(), right["count"].to_numpy()
    point = numpy.sqrt(ls.sum() / lc.sum()) - numpy.sqrt(rs.sum() / rc.sum())
    draws = RNG.integers(0, len(days), size=(RESAMPLES, len(days)))
    left_rmsd = numpy.sqrt(ls[draws].sum(axis=1) / lc[draws].sum(axis=1))
    right_rmsd = numpy.sqrt(rs[draws].sum(axis=1) / rc[draws].sum(axis=1))
    differences = left_rmsd - right_rmsd
    return point, float(numpy.percentile(differences, 2.5)), float(numpy.percentile(differences, 97.5))


lines = ["# CURRENT_TEST strata, challenger glonet, currents only", ""]
lines += [
    "Basis: FILTR minus wind slippage (the LWS rung), default policy row set, plus the",
    "confirmed undrogued stratum 011 restored from the source archive so it is scored on",
    "the same basis. One scoring pass, 52 start dates, region global, leads 1 to 9.",
    "RMSD and bias in m/s. uv pools the two components, so its count is twice the matchups.",
    "Bias is model minus observation.",
    "",
]

# --- global pooled -----------------------------------------------------------
component = pooled(frame, ["current_test", "variable"])
combined = pooled(frame, ["current_test"])
combined["variable"] = "uv"
lines += ["## Global pooled, leads 1 to 9", "",
          "| code | matchups | uo rmsd | vo rmsd | uv rmsd | uo bias | vo bias | uv vs 313 | 95% CI | scale 313=0 011=1 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
reference = float(combined[combined["current_test"] == 313]["rmsd"].iloc[0])
contaminated = float(combined[combined["current_test"] == 11]["rmsd"].iloc[0])
span = contaminated - reference
global_rows = {}
for code in CODES:
    uo = component[(component["current_test"] == code) & (component["variable"] == "uo")].iloc[0]
    vo = component[(component["current_test"] == code) & (component["variable"] == "vo")].iloc[0]
    uv = combined[combined["current_test"] == code].iloc[0]
    test = bootstrap_difference(frame, code)
    difference = "reference" if code == 313 else f"{test[0]:+.5f}"
    interval = "" if code == 313 else f"[{test[1]:+.5f}, {test[2]:+.5f}]"
    position = "" if code == 313 else f"{(float(uv['rmsd']) - reference) / span:.2f}"
    if code == 313:
        position = "0.00"
    global_rows[code] = (float(uv["rmsd"]), int(uo["count"]))
    lines.append("| " + " | ".join([
        code_label(code), str(int(uo["count"])),
        f"{float(uo['rmsd']):.5f}", f"{float(vo['rmsd']):.5f}", f"{float(uv['rmsd']):.5f}",
        f"{float(uo['bias']):+.5f}", f"{float(vo['bias']):+.5f}",
        difference, interval, position,
    ]) + " |")
lines.append("")

# --- lead 1 vs lead 9 --------------------------------------------------------
lines += ["## Lead 1 against lead 9, global, uv", "",
          "| code | n lead 1 | uv rmsd lead 1 | vs 313 | n lead 9 | uv rmsd lead 9 | vs 313 | growth |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
lead_tables = {}
for lead in (1, 9):
    lead_tables[lead] = pooled(frame[frame["lead"] == lead], ["current_test"]).set_index("current_test")
reference_lead = {lead: float(lead_tables[lead].loc[313, "rmsd"]) for lead in (1, 9)}
for code in CODES:
    first = lead_tables[1].loc[code]
    last = lead_tables[9].loc[code]
    delta_first = float(first["rmsd"]) - reference_lead[1]
    delta_last = float(last["rmsd"]) - reference_lead[9]
    lines.append("| " + " | ".join([
        code_label(code), str(int(first["count"] // 2)), f"{float(first['rmsd']):.5f}",
        f"{delta_first:+.5f}", str(int(last["count"] // 2)), f"{float(last['rmsd']):.5f}",
        f"{delta_last:+.5f}", f"{delta_last - delta_first:+.5f}",
    ]) + " |")
lines.append("")

# --- latitude bands ----------------------------------------------------------
band_table = pooled(frame, ["current_test", "lat_band"]).set_index(["current_test", "lat_band"])
lines += ["## Latitude bands, uv RMSD, leads 1 to 9", "",
          "Each cell is the uv RMSD and, in brackets, the matchup count.",
          "",
          "| band | " + " | ".join(code_label(c) for c in CODES) + " |",
          "| --- | " + " | ".join("---" for _ in CODES) + " |"]
band_records = []
for band in BANDS:
    cells = []
    for code in CODES:
        try:
            row = band_table.loc[(code, band)]
        except KeyError:
            cells.append("n/a")
            continue
        cells.append(f"{float(row['rmsd']):.4f} ({int(row['count'] // 2)})")
    lines.append(f"| {band} | " + " | ".join(cells) + " |")
lines.append("")

lines += ["### Band difference against 313, uv RMSD, with a 95 percent bootstrap interval over start dates", "",
          "| band | code | n | delta uv | 95% CI | distinguishable | scale 313=0 011=1 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
for band in BANDS:
    try:
        base = float(band_table.loc[(313, band), "rmsd"])
        worst = float(band_table.loc[(11, band), "rmsd"])
    except KeyError:
        continue
    for code in CODES:
        if code == 313:
            continue
        try:
            row = band_table.loc[(code, band)]
        except KeyError:
            continue
        if int(row["count"]) < 2000:
            continue
        test = bootstrap_difference(frame, code, band)
        if test is None:
            continue
        significant = "yes" if (test[1] > 0 or test[2] < 0) else "no"
        scale = (float(row["rmsd"]) - base) / (worst - base) if abs(worst - base) > 1e-9 else float("nan")
        band_records.append((band, code, float(row["rmsd"]) - base, test[1], test[2], significant, scale))
        lines.append("| " + " | ".join([
            str(band), code_label(code), str(int(row["count"] // 2)),
            f"{test[0]:+.5f}", f"[{test[1]:+.5f}, {test[2]:+.5f}]", significant, f"{scale:.2f}",
        ]) + " |")
lines.append("")

pandas.DataFrame(band_records, columns=["band", "code", "delta_uv", "ci_low", "ci_high", "significant", "scale"]).to_csv(
    os.path.join(OUT, "band_differences.csv"), index=False)
pooled(frame, ["current_test", "lat_band", "variable"]).to_csv(os.path.join(OUT, "per_code_band.csv"), index=False)
pooled(frame, ["current_test", "lead", "variable"]).to_csv(os.path.join(OUT, "per_code_lead.csv"), index=False)

with open(os.path.join(OUT, "RESULTS.md"), "w") as handle:
    handle.write("\n".join(lines) + "\n")
print("\n".join(lines))
