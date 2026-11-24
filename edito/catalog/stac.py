# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import base64
from enum import Enum

from registry import BucketStores, CatalogEntry, Collection, CopernicusMarineDatasets, TemporalExtent, bucket_url
from store_metadata import GLOBAL_SPATIAL_EXTENT, Dimension, DimensionKind, SpatialExtent, StoreMetadata, Variable

STAC_VERSION = "1.0.0"
CATALOG_IDENTIFIER = "oceanbench-catalog"
CATALOG_BASE_IDENTIFIER = "oceanbench"
PROJECT_GROUP = "oceanbench"
DEFAULT_GROUP = "default"
LICENSE = "other"
WEBSITE_URL = "https://oceanbench.edito.eu"
REPOSITORY_URL = "https://github.com/mercator-ocean/oceanbench"
LOGO_URL = bucket_url("public/logo/oceanbench-logo-light.png")
TIMELESS_DATETIME = "1970-01-01T00:00:00Z"
COPERNICUS_MARINE_PRODUCT_URL = "https://data.marine.copernicus.eu/product/{product_identifier}/description"
WGS84_EPSG_CODE = 4326
STAC_EXTENSIONS = (
    "https://stac-extensions.github.io/datacube/v2.2.0/schema.json",
    "https://stac-extensions.github.io/example-links/v0.0.1/schema.json",
)
PYTHON_MEDIA_TYPE = "text/x-python"
HTML_MEDIA_TYPE = "text/html"
ZARR_MEDIA_TYPE = "application/vnd+zarr"
ONE_DEGREE_SUFFIX = "_1_degree"
PROVIDERS = (
    {
        "name": "Mercator Ocean International",
        "description": "Producer and maintainer of OceanBench.",
        "roles": ["producer", "processor", "licensor"],
        "url": "https://www.mercator-ocean.eu",
    },
    {
        "name": "EDITO",
        "description": "European Digital Twin of the Ocean platform hosting the OceanBench bucket.",
        "roles": ["host"],
        "url": "https://dive.edito.eu",
    },
)
CATALOG_DESCRIPTION = (
    "OceanBench is a benchmarking tool to evaluate ocean forecasting systems against reference ocean analysis "
    "datasets and observations. This catalog gathers the datasets that the OceanBench library uses, in the "
    "OceanBench bucket and in Copernicus Marine, and how to open them."
    f"\n\n- [Website]({WEBSITE_URL})\n- [Source code]({REPOSITORY_URL})"
)
COLLECTION_DESCRIPTIONS = {
    Collection.INPUTS: "Initial conditions and forcings given to the challengers.",
    Collection.CHALLENGERS: "Forecasts of the ocean forecasting systems evaluated by OceanBench.",
    Collection.REFERENCES: "Analyses, reanalyses and observations the challengers are evaluated against.",
    Collection.TRAINING: "Datasets to train data-driven ocean forecasting systems.",
    Collection.AUXILIARY: "Static datasets used during the evaluation.",
}


class Visibility(Enum):
    PUBLIC = (DEFAULT_GROUP, PROJECT_GROUP)


def collection_identifier(collection: Collection) -> str:
    return f"{CATALOG_BASE_IDENTIFIER}-{collection.value}"


def stac_catalog() -> dict:
    return {
        "id": CATALOG_IDENTIFIER,
        "type": "Catalog",
        "stac_version": STAC_VERSION,
        "title": "OceanBench",
        "description": CATALOG_DESCRIPTION,
        "assets": {"thumbnail": {"href": LOGO_URL, "title": "Logo", "type": "image/png", "roles": ["thumbnail"]}},
        "visibility": list(Visibility.PUBLIC.value),
    }


def stac_collection(collection: Collection, entries: tuple[CatalogEntry, ...]) -> dict:
    temporal_extents = tuple(entry.temporal_extent for entry in entries if entry.temporal_extent)
    return {
        "id": collection_identifier(collection),
        "type": "Collection",
        "stac_version": STAC_VERSION,
        "title": f"OceanBench {collection.value}",
        "description": COLLECTION_DESCRIPTIONS[collection],
        "license": LICENSE,
        "keywords": ["OceanBench", "Ocean", "Forecast", collection.value],
        "providers": list(PROVIDERS),
        "extent": {
            "spatial": {"bbox": [_bounding_box(GLOBAL_SPATIAL_EXTENT)]},
            "temporal": {"interval": [_interval(_merged_temporal_extent(temporal_extents))]},
        },
        "links": _project_links(),
        "visibility": list(Visibility.PUBLIC.value),
    }


def stac_item(entry: CatalogEntry, metadata: StoreMetadata | None) -> dict:
    spatial_extent = metadata.spatial_extent if metadata else GLOBAL_SPATIAL_EXTENT
    variables = metadata.variables if metadata else ()
    dimensions = metadata.dimensions if metadata else ()
    return {
        "type": "Feature",
        "stac_version": STAC_VERSION,
        "stac_extensions": list(STAC_EXTENSIONS),
        "id": f"{CATALOG_BASE_IDENTIFIER}-{entry.identifier}",
        "collection": collection_identifier(entry.collection),
        "geometry": _polygon(spatial_extent),
        "bbox": _bounding_box(spatial_extent),
        "properties": {
            "title": entry.title,
            "description": entry.description,
            **_datetime_properties(entry.temporal_extent),
            "keywords": ["OceanBench", entry.collection.value, *_standard_names(variables)],
            "license": LICENSE,
            "providers": list(PROVIDERS),
            "oceanbench:library_functions": list(entry.library_functions),
            "cube:dimensions": {dimension.name: _cube_dimension(dimension) for dimension in dimensions},
            "cube:variables": {variable.name: _cube_variable(variable) for variable in variables},
            "visibility": list(Visibility.PUBLIC.value),
        },
        "assets": _assets(entry.source),
        "links": [*_example_links(entry, metadata), *_source_links(entry.source), *_project_links()],
    }


def _merged_temporal_extent(extents: tuple[TemporalExtent, ...]) -> TemporalExtent | None:
    match extents:
        case ():
            return None
        case _:
            return TemporalExtent(
                min(extent.first_day for extent in extents), max(extent.last_day for extent in extents)
            )


def _bounding_box(extent: SpatialExtent) -> list[float]:
    return [extent.west, extent.south, extent.east, extent.north]


def _polygon(extent: SpatialExtent) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [extent.west, extent.south],
                [extent.east, extent.south],
                [extent.east, extent.north],
                [extent.west, extent.north],
                [extent.west, extent.south],
            ]
        ],
    }


def _interval(extent: TemporalExtent | None) -> list[str | None]:
    return (
        [f"{extent.first_day.isoformat()}T00:00:00Z", f"{extent.last_day.isoformat()}T00:00:00Z"]
        if extent
        else [None, None]
    )


def _datetime_properties(extent: TemporalExtent | None) -> dict:
    start, end = _interval(extent) if extent else (TIMELESS_DATETIME, TIMELESS_DATETIME)
    return {"datetime": None, "start_datetime": start, "end_datetime": end}


def _standard_names(variables: tuple[Variable, ...]) -> list[str]:
    return list(dict.fromkeys(variable.standard_name for variable in variables if variable.standard_name))


def _cube_dimension(dimension: Dimension) -> dict:
    match dimension.kind:
        case DimensionKind.LONGITUDE:
            return {
                "type": "spatial",
                "axis": "x",
                "extent": list(dimension.extent),
                "reference_system": WGS84_EPSG_CODE,
            }
        case DimensionKind.LATITUDE:
            return {
                "type": "spatial",
                "axis": "y",
                "extent": list(dimension.extent),
                "reference_system": WGS84_EPSG_CODE,
            }
        case DimensionKind.TEMPORAL:
            return {"type": "temporal", "extent": list(dimension.extent)}
        case DimensionKind.OTHER:
            return {"type": dimension.name, "extent": list(dimension.extent)}


def _cube_variable(variable: Variable) -> dict:
    return {
        "type": "data",
        "dimensions": list(variable.dimensions),
        **({"unit": variable.units} if variable.units else {}),
        **({"description": variable.long_name} if variable.long_name else {}),
    }


def _assets(source: BucketStores | CopernicusMarineDatasets) -> dict:
    match source:
        case BucketStores(sample_path=sample_path):
            return {
                "sample": {
                    "href": bucket_url(sample_path),
                    "title": "Sample store",
                    "type": ZARR_MEDIA_TYPE,
                    "roles": ["data"],
                }
            }
        case CopernicusMarineDatasets(product_identifier=product_identifier):
            return {
                "product": {
                    "href": COPERNICUS_MARINE_PRODUCT_URL.format(product_identifier=product_identifier),
                    "title": "Copernicus Marine product",
                    "type": HTML_MEDIA_TYPE,
                    "roles": ["metadata"],
                }
            }


def _example_links(entry: CatalogEntry, metadata: StoreMetadata | None) -> list[dict]:
    match (entry.library_functions, entry.source):
        case ((), BucketStores(sample_path=sample_path)):
            return [_example_link("Open with xarray", _xarray_snippet(bucket_url(sample_path), metadata))]
        case ((), CopernicusMarineDatasets(dataset_identifiers=identifiers)):
            return [_example_link("Open with the Copernicus Marine toolbox", _copernicus_marine_snippet(identifiers))]
        case (functions, CopernicusMarineDatasets(dataset_identifiers=identifiers)):
            return [
                _example_link("Open with OceanBench", _library_snippet(functions)),
                _example_link("Open with the Copernicus Marine toolbox", _copernicus_marine_snippet(identifiers)),
            ]
        case (functions, _):
            return [_example_link("Open with OceanBench", _library_snippet(functions))]


def _library_snippet(functions: tuple[str, ...]) -> str:
    return "import oceanbench\n\n" + "".join(
        f"{_snippet_variable(function)} = {function}()\n" for function in functions
    )


def _snippet_variable(function: str) -> str:
    return "dataset_1_degree" if function.endswith(ONE_DEGREE_SUFFIX) else "dataset"


def _xarray_snippet(url: str, metadata: StoreMetadata | None) -> str:
    match metadata:
        case StoreMetadata(groups=groups) if groups:
            return (
                "import xarray\n\n"
                f'store_url = "{url}"\n'
                "dataset = xarray.merge(\n"
                f"    [xarray.open_zarr(store_url, group=group) for group in {list(groups)!r}],\n"
                '    compat="override",\n'
                '    join="override",\n'
                ")\n"
            )
        case _:
            return f'import xarray\n\ndataset = xarray.open_zarr("{url}")\n'


def _copernicus_marine_snippet(identifiers: tuple[str, ...]) -> str:
    match identifiers:
        case (identifier,):
            return f'import copernicusmarine\n\ndataset = copernicusmarine.open_dataset(dataset_id="{identifier}")\n'
        case _:
            return (
                "import copernicusmarine\nimport xarray\n\n"
                "dataset = xarray.merge(\n    [\n"
                + "".join(
                    f'        copernicusmarine.open_dataset(dataset_id="{identifier}"),\n' for identifier in identifiers
                )
                + "    ]\n)\n"
            )


def _example_link(title: str, snippet: str) -> dict:
    return {
        "rel": "example",
        "href": f"data:{PYTHON_MEDIA_TYPE};base64,{base64.b64encode(snippet.encode()).decode()}",
        "title": title,
        "type": PYTHON_MEDIA_TYPE,
        "example:language": "Python",
        "example:container": False,
    }


def _source_links(source: BucketStores | CopernicusMarineDatasets) -> list[dict]:
    match source:
        case CopernicusMarineDatasets(product_identifier=product_identifier):
            return [
                {
                    "rel": "about",
                    "href": COPERNICUS_MARINE_PRODUCT_URL.format(product_identifier=product_identifier),
                    "title": f"Copernicus Marine product {product_identifier}",
                    "type": HTML_MEDIA_TYPE,
                }
            ]
        case BucketStores():
            return []


def _project_links() -> list[dict]:
    return [
        {"rel": "about", "href": WEBSITE_URL, "title": "OceanBench website", "type": HTML_MEDIA_TYPE},
        {"rel": "related", "href": REPOSITORY_URL, "title": "OceanBench source code", "type": HTML_MEDIA_TYPE},
    ]
