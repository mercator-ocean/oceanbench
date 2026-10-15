# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from copy import deepcopy
from numbers import Real


def _average_score(scores: list[dict]) -> dict | None:
    result = deepcopy(scores[0])
    depths = {}
    for depth, first_depth in scores[0].get("depths", {}).items():
        if not all(depth in score.get("depths", {}) for score in scores[1:]):
            continue
        source_depths = [score["depths"][depth] for score in scores]
        variables = {}
        for variable, first_variable in first_depth.get("variables", {}).items():
            if not all(variable in source.get("variables", {}) for source in source_depths[1:]):
                continue
            source_variables = [source["variables"][variable] for source in source_depths]
            if any(
                source.get(field) != source_variables[0].get(field)
                for source in source_variables[1:]
                for field in ("unit", "standard_name")
            ):
                continue
            data = {}
            for lead_day in first_variable.get("data", {}):
                values = [source.get("data", {}).get(lead_day) for source in source_variables]
                if all(isinstance(value, Real) and not isinstance(value, bool) for value in values):
                    data[lead_day] = sum(values) / len(values)
            if data:
                averaged_variable = deepcopy(first_variable)
                averaged_variable["data"] = data
                variables[variable] = averaged_variable
        if variables:
            averaged_depth = deepcopy(first_depth)
            averaged_depth["variables"] = variables
            depths[depth] = averaged_depth
    if not depths:
        return None
    result["depths"] = depths
    return result


def averaged_version_bundle(bundles: dict[str, dict], versions: tuple[str, ...], label: str) -> dict | None:
    sources = [bundles.get(version) for version in versions]
    if any(source is None for source in sources):
        return None
    regions = {}
    for region_id in sources[0].get("region_order", []):
        if not all(region_id in source.get("regions", {}) for source in sources[1:]):
            continue
        source_regions = [source["regions"][region_id] for source in sources]
        challengers = {}
        for challenger in source_regions[0].get("challenger_names", []):
            if not all(challenger in region.get("challengers", {}) for region in source_regions[1:]):
                continue
            source_scores = [region["challengers"][challenger] for region in source_regions]
            metrics = {}
            for metric in source_scores[0]:
                if all(metric in score_set for score_set in source_scores[1:]):
                    score = _average_score([score_set[metric] for score_set in source_scores])
                    if score:
                        metrics[metric] = score
            if metrics:
                challengers[challenger] = metrics
        if challengers:
            regions[region_id] = {
                "display_name": source_regions[0].get("display_name", region_id),
                "challengers": challengers,
                "challenger_names": [name for name in source_regions[0]["challenger_names"] if name in challengers],
            }
    if not regions:
        return None
    region_order = list(regions)
    challenger_names = {name for region in regions.values() for name in region["challenger_names"]}
    return {
        "display_name": label,
        "source_versions": list(versions),
        "has_report_links": False,
        "regions": regions,
        "region_order": region_order,
        "region_labels": {region: sources[0]["region_labels"].get(region, region) for region in region_order},
        "region_metadata": {region: sources[0]["region_metadata"].get(region, {}) for region in region_order},
        "challenger_labels": {
            name: sources[0]["challenger_labels"].get(name, name)
            for region in regions.values()
            for name in region["challenger_names"]
        },
        "challenger_notes": {
            name: notes
            for source in sources
            for name, notes in source.get("challenger_notes", {}).items()
            if name in challenger_names
        },
        "challenger_categories": {
            name: category
            for source in sources
            for name, category in source.get("challenger_categories", {}).items()
            if name in challenger_names
        },
    }
