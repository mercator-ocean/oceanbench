# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from helpers.challenger_metadata import challenger_category, challenger_label, challenger_note
from helpers.published_regions import (
    published_region_ids_with_scores_files,
    published_region_label,
    published_region_metadata,
)
from helpers.s3_discovery import (
    DOWNLOADED_SCORES_DIRECTORY,
    available_versions,
    default_version,
    discover_downloaded_scores_files,
    downloaded_scores_file_path,
)
from helpers.scores_file_reader import get_all_model_scores_from_scores_file


def _load_scores(version: str, challenger_name: str, region_id: str, strict: bool) -> dict | None:
    local_path = downloaded_scores_file_path(DOWNLOADED_SCORES_DIRECTORY, version, challenger_name, region_id)
    if local_path is None:
        return None
    try:
        scores = get_all_model_scores_from_scores_file(local_path, challenger_name)
    except Exception as error:
        if strict:
            raise
        print(f"Skipping {version}/{challenger_name}.{region_id}: {error}")
        return None
    if not scores:
        if strict:
            raise RuntimeError(f"No scores parsed for {version}/{challenger_name}.{region_id}.")
        return None
    return {metric_key: score.model_dump() for metric_key, score in scores.items()}


def _region_bundle(version: str, region_id: str, challenger_names: list[str], strict: bool) -> dict | None:
    challengers = {
        challenger_name: scores
        for challenger_name in challenger_names
        if (scores := _load_scores(version, challenger_name, region_id, strict))
    }
    if not challengers:
        return None
    return {
        "display_name": published_region_label(region_id),
        "challengers": challengers,
        "challenger_names": list(challengers.keys()),
    }


def _version_bundle(version: str, strict: bool) -> dict:
    published_scores_files = discover_downloaded_scores_files(DOWNLOADED_SCORES_DIRECTORY, version)
    region_bundles = {
        region_id: region_bundle
        for region_id in published_region_ids_with_scores_files(published_scores_files)
        if (region_bundle := _region_bundle(version, region_id, published_scores_files[region_id], strict))
    }
    region_ids = list(region_bundles.keys())
    challenger_names = [
        challenger_name for region in region_bundles.values() for challenger_name in region["challenger_names"]
    ]
    return {
        "regions": region_bundles,
        "region_order": region_ids,
        "region_labels": {region_id: published_region_label(region_id) for region_id in region_ids},
        "region_metadata": {region_id: published_region_metadata(region_id) for region_id in region_ids},
        "challenger_labels": {
            challenger_name: challenger_label(challenger_name) for challenger_name in challenger_names
        },
        "challenger_notes": {
            challenger_name: note for challenger_name in challenger_names if (note := challenger_note(challenger_name))
        },
        "challenger_categories": {
            challenger_name: challenger_category(challenger_name) for challenger_name in challenger_names
        },
    }


def version_bundles() -> tuple[dict[str, dict], str]:
    """The scores of every published version that has some, and the version shown by default."""
    declared_default_version = default_version()
    bundles = {
        version: bundle
        for version in available_versions()
        if (bundle := _version_bundle(version, strict=version == declared_default_version))["region_order"]
    }
    resolved_default_version = (
        declared_default_version if declared_default_version in bundles else next(iter(bundles), "")
    )
    return bundles, resolved_default_version
