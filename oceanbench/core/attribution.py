# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Copernicus Marine attribution and disclaimer (contracts.md §11).

The derived-work credit is a strict superset of the plain-redistribution credit
(never wrong), so OceanBench uses it everywhere. The credit and the standard
CMEMS liability / no-warranty disclaimer are embedded in the zarr attrs of every
viewer pyramid and evaluation pack, pack READMEs, the viewer footer and the
website data-provenance page. ``copernicus_marine_attribution_attrs`` returns the
attrs mapping to splice into a root zarr group.
"""

COPERNICUS_MARINE_CREDIT = (
    "Generated using E.U. Copernicus Marine Service Information; "
    "https://doi.org/10.48670/moi-00021 ; https://doi.org/10.48670/moi-00016"
)

COPERNICUS_MARINE_DISCLAIMER = (
    "This is an OceanBench-generated derived product and is not the authoritative "
    "Copernicus Marine product. The E.U. Copernicus Marine Service Information is "
    "provided without any warranty, express or implied, including the warranties of "
    "merchantability and fitness for a particular purpose. Neither the European Union, "
    "Mercator Ocean International nor OceanBench is liable for any consequence stemming "
    "from the use of this derived product."
)

COPERNICUS_MARINE_SOURCE_PRODUCTS = {
    "GLORYS12": "GLOBAL_MULTIYEAR_PHY_001_030",
    "GLO12": "GLOBAL_ANALYSISFORECAST_PHY_001_024",
}


def copernicus_marine_attribution_attrs() -> dict[str, object]:
    """Attrs to embed in a viewer-pyramid / pack root zarr group (contracts.md §11)."""
    return {
        "attribution": COPERNICUS_MARINE_CREDIT,
        "disclaimer": COPERNICUS_MARINE_DISCLAIMER,
        "source_products": dict(COPERNICUS_MARINE_SOURCE_PRODUCTS),
        "derived_product": "OceanBench-generated",
    }
