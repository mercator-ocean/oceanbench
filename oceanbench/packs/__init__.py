# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Local evaluation against an offline reference directory (contracts.md §7).

``oceanbench evaluate`` reads its references live from the bucket by default. Pointing
``--offline-references`` at a local directory makes it read everything from disk instead:
the gridded references, the Class-4 observation match-up store, the mean-dynamic-topography
needed for the SSH->SLA conversion, and a ``pack-manifest.json`` stamping the upstream
products the directory derives from, its region and its year.
"""

from oceanbench.packs.manifest import PACK_MANIFEST_FILENAME, PackManifestResult, write_pack_manifest

__all__ = [
    "PACK_MANIFEST_FILENAME",
    "PackManifestResult",
    "write_pack_manifest",
]
