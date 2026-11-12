# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import base64
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath

from bucket import BucketObject, ObjectKind, listing_url, public_url, s3_uri
from dataset_metadata import (
    DatasetMetadata,
    Dimension,
    DimensionKind,
    MetadataUnavailable,
    SpatialExtent,
    TemporalExtent,
    Variable,
    merged_spatial_extents,
    merged_temporal_extents,
)
from dataset_resolution import BucketDataset, CopernicusMarineCatalogDataset, ResolvedDataset
from datasets import COPERNICUS_MARINE_COLLECTION, BucketObjects, DatasetDescription, LibraryUsage


class CatalogAudience(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"


STAC_VERSION = "1.0.0"
HEALTH_PROPERTY = "oceanbench:health"
CATALOG_IDENTIFIER = "oceanbench-catalog"
CATALOG_BASE_IDENTIFIER = "oceanbench"
PROJECT_GROUP = "oceanbench"
DEFAULT_GROUP = "default"
LICENSE = "other"
WEBSITE_URL = "https://oceanbench.edito.eu"
REPOSITORY_URL = "https://github.com/mercator-ocean/oceanbench"
LOGO_URL = public_url("public/logo/oceanbench-logo-light.png")
COPERNICUS_MARINE_PRODUCT_URL = "https://data.marine.copernicus.eu/product/{product_identifier}/description"
DIGITAL_OBJECT_IDENTIFIER_URL = "https://doi.org/{digital_object_identifier}"
WGS84_EPSG_CODE = 4326
GLOBAL_BOUNDING_BOX = SpatialExtent(west=-180.0, south=-90.0, east=180.0, north=90.0)
STAC_EXTENSIONS = (
    "https://stac-extensions.github.io/datacube/v2.2.0/schema.json",
    "https://stac-extensions.github.io/example-links/v0.0.1/schema.json",
    "https://stac-extensions.github.io/scientific/v1.0.0/schema.json",
)
MEDIA_TYPES = {
    ".zarr": "application/vnd+zarr",
    ".nc": "application/x-netcdf",
    ".csv": "text/csv",
    ".parquet": "application/vnd.apache.parquet",
    ".tar": "application/x-tar",
    ".dat": "text/plain",
    ".jsonl": "application/x-ndjson",
}
DEFAULT_MEDIA_TYPE = "application/octet-stream"
PYTHON_MEDIA_TYPE = "text/x-python"
PUBLIC_API_PREFIX = "oceanbench.datasets."
XML_MEDIA_TYPE = "application/xml"
HTML_MEDIA_TYPE = "text/html"
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
CATALOG_DESCRIPTIONS = {
    CatalogAudience.PUBLIC: (
        "OceanBench is a benchmarking tool to evaluate ocean forecasting systems against reference ocean analysis "
        "datasets and observations. This catalog gathers the datasets that the OceanBench library uses, in the "
        "OceanBench bucket and in Copernicus Marine, and how to open them."
        f"\n\n- [Website]({WEBSITE_URL})\n- [Source code]({REPOSITORY_URL})"
    ),
    CatalogAudience.INTERNAL: (
        "Every dataset of the OceanBench bucket, whether the OceanBench library uses it or not, and the Copernicus "
        "Marine datasets it relies on, with their storage locations and examples to open them. "
        "For the OceanBench maintainers."
        f"\n\n- [Website]({WEBSITE_URL})\n- [Source code]({REPOSITORY_URL})"
    ),
}


class Visibility(Enum):
    PUBLIC = (DEFAULT_GROUP, PROJECT_GROUP)
    PROJECT = (PROJECT_GROUP,)


class StorageDetails(Enum):
    SHOWN = "shown"
    HIDDEN = "hidden"


@dataclass(frozen=True)
class CatalogEnvironment:
    name: str | None
    audience: CatalogAudience = CatalogAudience.PUBLIC

    def labels(self) -> tuple[str, ...]:
        audience_labels = (self.audience.value,) if self.audience == CatalogAudience.INTERNAL else ()
        name_labels = (self.name,) if self.name is not None else ()
        return (*audience_labels, *name_labels)

    def identifier(self, base_identifier: str) -> str:
        return "-".join((base_identifier, *map(str.lower, self.labels())))

    def visibility(self) -> Visibility:
        is_public = self.name is None and self.audience == CatalogAudience.PUBLIC
        return Visibility.PUBLIC if is_public else Visibility.PROJECT

    def storage(self) -> StorageDetails:
        return StorageDetails.HIDDEN if self.audience == CatalogAudience.PUBLIC else StorageDetails.SHOWN

    def title_suffix(self) -> str:
        return f" ({', '.join(self.labels())})" if self.labels() else ""

    def with_audience(self, audience: CatalogAudience) -> "CatalogEnvironment":
        return replace(self, audience=audience)


def catalog_identifier(environment: CatalogEnvironment) -> str:
    return environment.identifier(CATALOG_IDENTIFIER)


def catalog_datasets(
    datasets: tuple[ResolvedDataset, ...], environment: CatalogEnvironment
) -> tuple[ResolvedDataset, ...]:
    if environment.audience == CatalogAudience.INTERNAL:
        return datasets
    return tuple(dataset for dataset in datasets if dataset.description.library_usage == LibraryUsage.USED)


def collection_identifier(collection: str, environment: CatalogEnvironment) -> str:
    return environment.identifier(f"{CATALOG_BASE_IDENTIFIER}-{collection}")


def item_identifier(dataset: ResolvedDataset, environment: CatalogEnvironment) -> str:
    return environment.identifier(f"{CATALOG_BASE_IDENTIFIER}-{dataset.description.identifier}")


def stac_catalog(environment: CatalogEnvironment, health: dict | None = None) -> dict:
    return {
        "id": catalog_identifier(environment),
        "type": "Catalog",
        "stac_version": STAC_VERSION,
        "title": f"OceanBench{environment.title_suffix()}",
        "description": CATALOG_DESCRIPTIONS[environment.audience],
        "assets": {"thumbnail": {"href": LOGO_URL, "title": "Logo", "type": "image/png", "roles": ["thumbnail"]}},
        "visibility": list(Visibility.PROJECT.value),
        **({HEALTH_PROPERTY: health} if health is not None else {}),
    }


def stac_collection(collection: str, datasets: tuple[ResolvedDataset, ...], environment: CatalogEnvironment) -> dict:
    metadata = tuple(map(_metadata, datasets))
    spatial_extent = merged_spatial_extents(tuple(map(_spatial_extent, metadata))) or GLOBAL_BOUNDING_BOX
    temporal_extent = merged_temporal_extents(tuple(map(_dataset_temporal_extent, datasets)))
    return {
        "id": collection_identifier(collection, environment),
        "type": "Collection",
        "stac_version": STAC_VERSION,
        "title": f"OceanBench {_collection_title(collection, environment)}{environment.title_suffix()}",
        "description": _collection_description(collection, environment),
        "license": LICENSE,
        "keywords": ["OceanBench", "Ocean", "Forecast", collection],
        "providers": list(PROVIDERS),
        "extent": {
            "spatial": {"bbox": [_bounding_box(spatial_extent)]},
            "temporal": {"interval": [_interval(temporal_extent)]},
        },
        "links": _project_links(),
        "visibility": list(environment.visibility().value),
    }


def _collection_title(collection: str, environment: CatalogEnvironment) -> str:
    match (collection, environment.audience):
        case (name, _) if name == COPERNICUS_MARINE_COLLECTION:
            return "Copernicus Marine datasets"
        case _:
            return f"{collection} bucket datasets"


def _collection_description(collection: str, environment: CatalogEnvironment) -> str:
    match (collection, environment.audience):
        case (name, _) if name == COPERNICUS_MARINE_COLLECTION:
            return "Copernicus Marine datasets that the OceanBench library opens."
        case (_, CatalogAudience.PUBLIC):
            return "Datasets that the OceanBench library opens from its bucket."
        case _:
            return f"Datasets found in the `{collection}` directory of the OceanBench bucket."


def stac_item(dataset: ResolvedDataset, environment: CatalogEnvironment, publication_datetime: datetime) -> dict:
    metadata = _metadata(dataset)
    spatial_extent = _spatial_extent(metadata)
    storage = environment.storage()
    return {
        "type": "Feature",
        "stac_version": STAC_VERSION,
        "stac_extensions": list(STAC_EXTENSIONS),
        "id": item_identifier(dataset, environment),
        "geometry": _polygon(spatial_extent) if spatial_extent else None,
        **({"bbox": _bounding_box(spatial_extent)} if spatial_extent else {}),
        "properties": {
            "title": _item_title(dataset.description, environment),
            "description": _item_description(dataset, storage),
            **_datetime_properties(_dataset_temporal_extent(dataset), dataset, publication_datetime),
            "keywords": ["OceanBench", dataset.description.collection, *_standard_names(metadata)],
            "license": LICENSE,
            "providers": list(PROVIDERS),
            "oceanbench:library_usage": dataset.description.library_usage.value,
            **_library_function_property(dataset),
            **_source_properties(dataset, storage),
            **_datacube_properties(metadata),
            "visibility": list(environment.visibility().value),
        },
        "assets": _assets(dataset, storage),
        "links": [*_example_links(dataset, metadata, storage), *_source_links(dataset), *_project_links()],
    }


def _item_title(description: DatasetDescription, environment: CatalogEnvironment) -> str:
    match (environment.audience, description.library_usage):
        case (CatalogAudience.PUBLIC, _):
            return _public_title(description)
        case (CatalogAudience.INTERNAL, LibraryUsage.USED):
            return f"{description.title} (used)"
        case (CatalogAudience.INTERNAL, LibraryUsage.UNUSED):
            return f"{description.title} (not used)"


def _public_title(description: DatasetDescription) -> str:
    match description.source:
        case BucketObjects():
            return description.title.split("/", 1)[-1]
        case _:
            return description.title


def _metadata(dataset: ResolvedDataset) -> DatasetMetadata | MetadataUnavailable:
    match dataset:
        case BucketDataset(metadata=metadata):
            return metadata
        case CopernicusMarineCatalogDataset(datasets=MetadataUnavailable() as unavailable):
            return unavailable
        case CopernicusMarineCatalogDataset(datasets=copernicus_marine_datasets):
            return _merged_copernicus_marine_metadata(tuple(item.metadata for item in copernicus_marine_datasets))


def _merged_copernicus_marine_metadata(metadata: tuple[DatasetMetadata, ...]) -> DatasetMetadata:
    return DatasetMetadata(
        spatial_extent=merged_spatial_extents(tuple(item.spatial_extent for item in metadata)),
        temporal_extent=merged_temporal_extents(tuple(item.temporal_extent for item in metadata)),
        dimensions=tuple({dimension.name: dimension for item in metadata for dimension in item.dimensions}.values()),
        variables=tuple({variable.name: variable for item in metadata for variable in item.variables}.values()),
        groups=(),
    )


def _dataset_temporal_extent(dataset: ResolvedDataset) -> TemporalExtent | None:
    return _temporal_extent(_metadata(dataset))


def _spatial_extent(metadata: DatasetMetadata | MetadataUnavailable) -> SpatialExtent | None:
    return metadata.spatial_extent if isinstance(metadata, DatasetMetadata) else None


def _temporal_extent(metadata: DatasetMetadata | MetadataUnavailable) -> TemporalExtent | None:
    return metadata.temporal_extent if isinstance(metadata, DatasetMetadata) else None


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
    return [extent.start.isoformat(), extent.end.isoformat()] if extent else [None, None]


def _datetime_properties(
    extent: TemporalExtent | None, dataset: ResolvedDataset, publication_datetime: datetime
) -> dict:
    if extent is not None:
        return {"datetime": None, "start_datetime": extent.start.isoformat(), "end_datetime": extent.end.isoformat()}
    return {"datetime": (_last_modified(dataset) or publication_datetime).isoformat()}


def _last_modified(dataset: ResolvedDataset) -> datetime | None:
    match dataset:
        case BucketDataset(objects=objects):
            return max((item.last_modified for item in objects if item.last_modified), default=None)
        case _:
            return None


def _item_description(dataset: ResolvedDataset, storage: StorageDetails) -> str:
    usage = _usage_description(dataset.description)
    unavailable = _metadata(dataset)
    metadata_note = (
        f"\n\nMetadata could not be read: {unavailable.reason}"
        if isinstance(unavailable, MetadataUnavailable) and storage == StorageDetails.SHOWN
        else ""
    )
    return f"{usage}{_source_description(dataset, storage)}{metadata_note}"


def _usage_description(description: DatasetDescription) -> str:
    match description.library_usage:
        case LibraryUsage.UNUSED:
            return "Not used by the OceanBench library."
        case LibraryUsage.USED:
            functions = "".join(f"\n- `{function}`" for function in description.library_functions)
            return "Used by the OceanBench library." + (
                f" Library functions leading to it:{functions}" if functions else ""
            )


def _source_description(dataset: ResolvedDataset, storage: StorageDetails) -> str:
    match dataset:
        case BucketDataset() if storage == StorageDetails.HIDDEN:
            return ""
        case BucketDataset(objects=objects):
            return (
                f"\n\n{len(objects)} object(s) in `{s3_uri(_common_directory(objects))}`, "
                f"from `{_name(objects[0])}` to `{_name(objects[-1])}`."
            )
        case CopernicusMarineCatalogDataset(datasets=MetadataUnavailable()):
            return ""
        case CopernicusMarineCatalogDataset(datasets=copernicus_marine_datasets):
            return "\n\n" + "\n".join(
                f"- `{item.dataset_identifier}` from [{item.product_title}]"
                f"({COPERNICUS_MARINE_PRODUCT_URL.format(product_identifier=item.product_identifier)})"
                for item in copernicus_marine_datasets
            )


def _common_directory(objects: tuple[BucketObject, ...]) -> str:
    directories = tuple(PurePosixPath(item.key).parent.parts for item in objects)
    return "/".join(directories[0][: _common_prefix_length(directories)]) + "/"


def _common_prefix_length(directories: tuple[tuple[str, ...], ...]) -> int:
    return next(
        (index for index, parts in enumerate(zip(*directories)) if any(part != parts[0] for part in parts)),
        min(map(len, directories)),
    )


def _name(bucket_object: BucketObject) -> str:
    return PurePosixPath(bucket_object.key).name


def _library_function_property(dataset: ResolvedDataset) -> dict:
    functions = dataset.description.library_functions
    return {"oceanbench:library_functions": list(functions)} if functions else {}


def _source_properties(dataset: ResolvedDataset, storage: StorageDetails) -> dict:
    match dataset:
        case BucketDataset(objects=objects):
            sizes = tuple(item.size_in_bytes for item in objects if item.size_in_bytes is not None)
            return {
                "oceanbench:object_count": len(objects),
                **(
                    {
                        "oceanbench:first_object": s3_uri(objects[0].key),
                        "oceanbench:last_object": s3_uri(objects[-1].key),
                    }
                    if storage == StorageDetails.SHOWN
                    else {}
                ),
                **({"file:size": sum(sizes)} if len(sizes) == len(objects) else {}),
            }
        case CopernicusMarineCatalogDataset(datasets=MetadataUnavailable()):
            return {}
        case CopernicusMarineCatalogDataset(datasets=copernicus_marine_datasets):
            return {
                "copernicusmarine:dataset_identifiers": [
                    item.dataset_identifier for item in copernicus_marine_datasets
                ],
                "sci:publications": [
                    {"doi": item.digital_object_identifier, "citation": item.product_title}
                    for item in copernicus_marine_datasets
                    if item.digital_object_identifier
                ],
            }


def _standard_names(metadata: DatasetMetadata | MetadataUnavailable) -> list[str]:
    if isinstance(metadata, MetadataUnavailable):
        return []
    return list(dict.fromkeys(variable.standard_name for variable in metadata.variables if variable.standard_name))


def _datacube_properties(metadata: DatasetMetadata | MetadataUnavailable) -> dict:
    if isinstance(metadata, MetadataUnavailable):
        return {}
    return {
        "cube:dimensions": {dimension.name: _cube_dimension(dimension) for dimension in metadata.dimensions},
        "cube:variables": {variable.name: _cube_variable(variable) for variable in metadata.variables},
        "cf:parameter": [
            {"name": variable.standard_name, "unit": variable.units}
            for variable in {
                variable.standard_name: variable for variable in metadata.variables if variable.standard_name
            }.values()
        ],
    }


def _cube_dimension(dimension: Dimension) -> dict:
    extent = (
        {"extent": [dimension.minimum, dimension.maximum]}
        if dimension.minimum is not None and dimension.maximum is not None
        else {}
    )
    step = {"step": dimension.step} if dimension.step is not None else {}
    size = {"description": f"{dimension.size} values"}
    match dimension.kind:
        case DimensionKind.LONGITUDE:
            return {"type": "spatial", "axis": "x", "reference_system": WGS84_EPSG_CODE, **extent, **step, **size}
        case DimensionKind.LATITUDE:
            return {"type": "spatial", "axis": "y", "reference_system": WGS84_EPSG_CODE, **extent, **step, **size}
        case DimensionKind.DEPTH:
            return {"type": "spatial", "axis": "z", "unit": "m", **extent, **step, **size}
        case DimensionKind.TIME:
            return {"type": "temporal", **extent, **size}
        case DimensionKind.OTHER:
            return {"type": dimension.name, **extent, **step, **size}


def _cube_variable(variable: Variable) -> dict:
    return {
        "type": "data",
        "dimensions": list(variable.dimensions),
        **({"unit": variable.units} if variable.units else {}),
        **({"description": variable.long_name} if variable.long_name else {}),
        **({"standardName": variable.standard_name} if variable.standard_name else {}),
    }


def _media_type(bucket_object: BucketObject) -> str:
    return MEDIA_TYPES.get(PurePosixPath(bucket_object.key).suffix, DEFAULT_MEDIA_TYPE)


def _assets(dataset: ResolvedDataset, storage: StorageDetails) -> dict:
    match dataset:
        case BucketDataset() if storage == StorageDetails.HIDDEN:
            return {}
        case BucketDataset(objects=objects):
            return {
                "latest": _object_asset(objects[-1], "Latest"),
                **({"first": _object_asset(objects[0], "First")} if len(objects) > 1 else {}),
                "listing": {
                    "href": listing_url(_common_directory(objects)),
                    "title": "Bucket listing",
                    "description": "S3 listing of the directory holding the dataset objects.",
                    "type": XML_MEDIA_TYPE,
                    "roles": ["metadata"],
                },
            }
        case CopernicusMarineCatalogDataset(datasets=MetadataUnavailable()):
            return {}
        case CopernicusMarineCatalogDataset(datasets=copernicus_marine_datasets):
            return {
                **{
                    item.dataset_identifier: {
                        "href": item.arco_url,
                        "title": f"{item.dataset_identifier} ARCO time series",
                        "type": MEDIA_TYPES[".zarr"],
                        "roles": ["data"],
                    }
                    for item in copernicus_marine_datasets
                    if item.arco_url and storage == StorageDetails.SHOWN
                },
                **{
                    "thumbnail": {
                        "href": item.thumbnail_url,
                        "title": item.product_title,
                        "type": "image/jpeg",
                        "roles": ["thumbnail"],
                    }
                    for item in copernicus_marine_datasets[:1]
                    if item.thumbnail_url
                },
            }


def _object_asset(bucket_object: BucketObject, position: str) -> dict:
    return {
        "href": public_url(bucket_object.key),
        "title": f"{position} {_name(bucket_object)}",
        "type": _media_type(bucket_object),
        "roles": ["data"],
        **({"file:size": bucket_object.size_in_bytes} if bucket_object.size_in_bytes is not None else {}),
        "alternate": {"s3": {"href": s3_uri(bucket_object.key)}},
    }


def _example_links(
    dataset: ResolvedDataset, metadata: DatasetMetadata | MetadataUnavailable, storage: StorageDetails
) -> list[dict]:
    source_example = _source_snippet(dataset, metadata, storage)
    return [
        *(
            _example_link(title, snippet)
            for title, snippet in map(_library_example, _shown_library_functions(dataset.description, storage))
        ),
        *([_example_link(*source_example)] if source_example else []),
    ]


def _shown_library_functions(description: DatasetDescription, storage: StorageDetails) -> tuple[str, ...]:
    match storage:
        case StorageDetails.SHOWN:
            return description.library_functions
        case StorageDetails.HIDDEN:
            return tuple(function for function in description.library_functions if _is_public_api(function))


def _is_public_api(function: str) -> bool:
    return function.startswith(PUBLIC_API_PREFIX)


def _library_example(function: str) -> tuple[str, str]:
    module, _, name = function.rpartition(".")
    match _is_public_api(function):
        case True:
            return (
                f"Open with oceanbench.datasets.{function.removeprefix(PUBLIC_API_PREFIX)}",
                f"import oceanbench\n\ndataset = {function}()\n",
            )
        case False:
            return (f"Library function {function}", f"from {module} import {name}\n")


def _example_link(title: str, snippet: str) -> dict:
    return {
        "rel": "example",
        "href": f"data:{PYTHON_MEDIA_TYPE};base64,{base64.b64encode(snippet.encode()).decode()}",
        "title": title,
        "type": PYTHON_MEDIA_TYPE,
        "example:language": "Python",
        "example:container": False,
    }


def _source_snippet(
    dataset: ResolvedDataset, metadata: DatasetMetadata | MetadataUnavailable, storage: StorageDetails
) -> tuple[str, str] | None:
    match dataset:
        case BucketDataset() if storage == StorageDetails.HIDDEN:
            return None
        case BucketDataset(objects=objects):
            return _bucket_object_snippet(objects[-1], metadata)
        case CopernicusMarineCatalogDataset(description=description):
            return ("Open with the Copernicus Marine toolbox", _copernicus_marine_snippet(description.source))


def _bucket_object_snippet(
    bucket_object: BucketObject, metadata: DatasetMetadata | MetadataUnavailable
) -> tuple[str, str] | None:
    url = public_url(bucket_object.key)
    suffix = PurePosixPath(bucket_object.key).suffix
    groups = metadata.groups if isinstance(metadata, DatasetMetadata) else ()
    match (bucket_object.kind, suffix, groups):
        case (ObjectKind.ZARR_STORE, _, ()):
            return ("Open the latest store with xarray", f'import xarray\n\ndataset = xarray.open_zarr("{url}")\n')
        case (ObjectKind.ZARR_STORE, _, _):
            return ("Open the latest store with xarray", _grouped_zarr_snippet(url, groups))
        case (_, ".nc", _):
            return (
                "Open the latest file with xarray",
                f'import xarray\n\ndataset = xarray.open_dataset("{url}#mode=bytes", engine="netcdf4", chunks={{}})\n',
            )
        case (_, ".csv", _):
            return ("Open the latest file with pandas", f'import pandas\n\ndataframe = pandas.read_csv("{url}")\n')
        case (_, ".parquet", _):
            return ("Open the latest file with pandas", f'import pandas\n\ndataframe = pandas.read_parquet("{url}")\n')
        case _:
            return None


def _grouped_zarr_snippet(url: str, groups: tuple[str, ...]) -> str:
    return (
        "import xarray\n\n"
        f'store_url = "{url}"\n'
        "dataset = xarray.merge(\n"
        f"    [xarray.open_zarr(store_url, group=group) for group in {groups!r}],\n"
        '    compat="override",\n'
        '    join="override",\n'
        ")\n"
    )


def _copernicus_marine_snippet(source) -> str:
    return "import copernicusmarine\n\n" + "".join(
        f'dataset = copernicusmarine.open_dataset(dataset_id="{dataset_identifier}")\n'
        for dataset_identifier in source.dataset_identifiers
    )


def _source_links(dataset: ResolvedDataset) -> list[dict]:
    match dataset:
        case CopernicusMarineCatalogDataset(datasets=MetadataUnavailable()):
            return []
        case CopernicusMarineCatalogDataset(datasets=copernicus_marine_datasets):
            products = {item.product_identifier: item for item in copernicus_marine_datasets}.values()
            return [
                *(
                    {
                        "rel": "about",
                        "href": COPERNICUS_MARINE_PRODUCT_URL.format(product_identifier=item.product_identifier),
                        "title": item.product_title,
                        "type": HTML_MEDIA_TYPE,
                    }
                    for item in products
                ),
                *(
                    {
                        "rel": "cite-as",
                        "href": DIGITAL_OBJECT_IDENTIFIER_URL.format(
                            digital_object_identifier=item.digital_object_identifier
                        ),
                        "title": f"{item.product_title} DOI",
                    }
                    for item in products
                    if item.digital_object_identifier
                ),
            ]
        case _:
            return []


def _project_links() -> list[dict]:
    return [
        {"rel": "about", "href": WEBSITE_URL, "title": "OceanBench website", "type": HTML_MEDIA_TYPE},
        {"rel": "related", "href": REPOSITORY_URL, "title": "OceanBench source code", "type": HTML_MEDIA_TYPE},
    ]
