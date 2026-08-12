# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import re
p = "/scratch/jseillade/obs-rebuild/rescore/aggregate_ladder.py"
s = open(p).read()
s = s.replace(
    'RUNG_ORDER = ["L0", "L1", "L2", "L2b", "L3"]',
    'RUNG_ORDER = ["L0", "L1", "L2", "L2b", "L3", "L1raw"]\n'
    '# the attribution chain, per variable. L1raw is off-chain and is compared to L1.\n'
    'CHAIN = {\n'
    '    "temperature": ["L0", "L1", "L3"],\n'
    '    "salinity": ["L0", "L1", "L3"],\n'
    '    "sla": ["L0", "L1", "L3"],\n'
    '    "uo": ["L0", "L1", "L2", "L2b", "L3"],\n'
    '    "vo": ["L0", "L1", "L2", "L2b", "L3"],\n'
    '}\n'
    'OFF_CHAIN = {"L1raw": "L1"}'
)
old = """    present = [r for r in RUNG_ORDER if r in set(group["rung"])]
    by_rung = group.set_index("rung")
    previous = None
    for rung in present:
        rmsd = float(by_rung.loc[rung, "rmsd"])
        count = int(by_rung.loc[rung, "count"])
        delta = numpy.nan if previous is None else rmsd - previous[1]
        percent = numpy.nan if previous is None else 100.0 * (rmsd - previous[1]) / previous[1]
        rows.append(
            {
                "region": region, "variable": variable, "lead": lead, "rung": rung,
                "rmsd": rmsd, "count": count,
                "previous_rung": None if previous is None else previous[0],
                "delta": delta, "percent": percent,
            }
        )
        previous = (rung, rmsd)"""
new = """    available = set(group["rung"])
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
    rows.extend(emitted)"""
assert old in s
s = s.replace(old, new)
open(p, "w").write(s)
print("patched")
