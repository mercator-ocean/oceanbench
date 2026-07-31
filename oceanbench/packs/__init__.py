# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Evaluation packs (contracts.md §7): downloadable, versioned reference bundles.

An evaluation pack is a self-describing directory produced by ``ingest`` from staged
reference / observation data so a model can be scored locally with
``oceanbench evaluate``. The pack carries the gridded references, the Class-4
observation match-up store, the mean-dynamic-topography needed for the SSH->SLA
conversion, a ``pack-manifest.json`` stamping the upstream products it derives from,
and a ``README.md`` with the Copernicus Marine credit and disclaimer (contracts.md §11).
"""

from oceanbench.packs.builder import DEFAULT_BASELINES, PackSources, build_pack
from oceanbench.packs.fetch import (
    PACK_FILE_INDEX_FILENAME,
    PackFetchSummary,
    default_pack_cache_directory,
    fetch_pack,
    resolve_offline_references,
    write_pack_file_index,
)
from oceanbench.packs.manifest import PACK_MANIFEST_FILENAME, PackManifestResult, write_pack_manifest

__all__ = [
    "DEFAULT_BASELINES",
    "PACK_FILE_INDEX_FILENAME",
    "PACK_MANIFEST_FILENAME",
    "PackFetchSummary",
    "PackManifestResult",
    "PackSources",
    "build_pack",
    "default_pack_cache_directory",
    "fetch_pack",
    "resolve_offline_references",
    "write_pack_file_index",
    "write_pack_manifest",
]
