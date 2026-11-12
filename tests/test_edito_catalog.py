# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import numpy
import pytest
import pandas
import xarray

CATALOG_DIRECTORY = Path(__file__).resolve().parents[1] / "edito" / "catalog"
sys.path.insert(0, str(CATALOG_DIRECTORY))

import edito_data_api  # noqa: E402
from bucket import BucketObject, ObjectKind  # noqa: E402
from catalog_health import (  # noqa: E402
    MAXIMUM_DELETIONS_PER_COLLECTION,
    BucketObjectCount,
    CatalogHealth,
    collection_changes,
    health_metadata,
    public_health_metadata,
    reusable_bucket_object_count,
    stored_bucket_object_count,
)
from dataset_metadata import CopernicusMarineDataset, dataset_metadata, merged_metadata  # noqa: E402
from dataset_resolution import BucketDataset, CopernicusMarineCatalogDataset  # noqa: E402
from datasets import (
    BucketObjects,
    CopernicusMarineDatasets,
    DatasetDescription,
    LibraryUsage,
    dataset_descriptions,
)  # noqa: E402
from library_usage import library_usage  # noqa: E402
from stac import (  # noqa: E402
    CatalogAudience,
    CatalogEnvironment,
    catalog_datasets,
    stac_catalog,
    stac_collection,
    stac_item,
)

LEAD_DAYS_COUNT = 7
PRODUCTION = CatalogEnvironment(None)
INTERNAL = CatalogEnvironment(None, CatalogAudience.INTERNAL)
PUBLICATION_DATETIME = datetime(2026, 10, 30, tzinfo=timezone.utc)


def _weekly_forecast(first_day: str, longitudes: list[float], latitudes: list[float]) -> xarray.Dataset:
    return xarray.Dataset(
        {
            "thetao": (
                ("first_day_datetime", "lead_day_index", "lat", "lon"),
                numpy.zeros((1, LEAD_DAYS_COUNT, len(latitudes), len(longitudes))),
                {"standard_name": "sea_water_potential_temperature", "units": "degrees_C"},
            )
        },
        coords={
            "first_day_datetime": pandas.to_datetime([first_day]),
            "lead_day_index": numpy.arange(LEAD_DAYS_COUNT),
            "lat": latitudes,
            "lon": longitudes,
        },
    )


def _bucket_dataset(
    identifier: str,
    first_day: str,
    last_day: str,
    west: float,
    east: float,
    library_usage: LibraryUsage = LibraryUsage.UNUSED,
    library_functions: tuple[str, ...] = (),
) -> BucketDataset:
    first_store = _weekly_forecast(first_day, [west, 0.0], [-10.0, 10.0])
    last_store = _weekly_forecast(last_day, [0.0, east], [-10.0, 10.0])
    return BucketDataset(
        description=DatasetDescription(
            identifier,
            f"public/{identifier}",
            "public",
            BucketObjects(f"public/{identifier}/"),
            library_usage,
            library_functions,
        ),
        objects=tuple(
            BucketObject(f"public/{identifier}/{day.replace('-', '')}.zarr", ObjectKind.ZARR_STORE, None, None)
            for day in (first_day, last_day)
        ),
        metadata=merged_metadata(dataset_metadata(first_store, ()), dataset_metadata(last_store, ())),
    )


def test_forecast_item_spans_from_first_forecast_day_to_last_lead_day():
    item = stac_item(
        _bucket_dataset("forecasts", "2024-01-03", "2024-12-25", -180.0, 179.75), PRODUCTION, PUBLICATION_DATETIME
    )

    assert item["properties"]["start_datetime"] == "2024-01-03T00:00:00+00:00"
    assert item["properties"]["end_datetime"] == "2024-12-31T00:00:00+00:00"
    assert item["bbox"] == [-180.0, -10.0, 179.75, 10.0]
    assert item["properties"]["cf:parameter"] == [{"name": "sea_water_potential_temperature", "unit": "degrees_C"}]


def test_collection_extent_covers_all_its_items():
    collection = stac_collection(
        "public",
        (
            _bucket_dataset("forecasts-2023", "2023-01-04", "2023-12-27", -180.0, 0.0),
            _bucket_dataset("forecasts-2024", "2024-01-03", "2024-12-25", -90.0, 179.75),
        ),
        PRODUCTION,
    )

    assert collection["extent"]["spatial"]["bbox"] == [[-180.0, -10.0, 179.75, 10.0]]
    assert collection["extent"]["temporal"]["interval"] == [["2023-01-04T00:00:00+00:00", "2024-12-31T00:00:00+00:00"]]


def test_few_vanished_datasets_are_deleted_from_the_catalog():
    changes = collection_changes("challengers", ("glonet", "xihe", "wenhai"), ("glonet", "xihe", "langya"))

    assert changes.added_items == ("wenhai",)
    assert changes.deleted_items == ("langya",)
    assert changes.withheld_deletions == ()


def test_mass_disappearance_is_withheld_instead_of_deleted():
    published = tuple(f"dataset-{index}" for index in range(MAXIMUM_DELETIONS_PER_COLLECTION + 2))

    changes = collection_changes("challengers", published[:1], published)

    assert changes.deleted_items == ()
    assert changes.withheld_deletions == published[1:]


def test_public_catalog_only_holds_datasets_used_by_the_library():
    used = _bucket_dataset("used", "2024-01-03", "2024-12-25", -180.0, 179.75, LibraryUsage.USED)
    unused = _bucket_dataset("unused", "2023-01-04", "2023-12-27", -90.0, 90.0)

    assert catalog_datasets((used, unused), PRODUCTION) == (used,)
    assert catalog_datasets((used, unused), INTERNAL) == (used, unused)


def test_public_catalog_items_and_collections_are_public():
    used = _bucket_dataset("used", "2024-01-03", "2024-12-25", -180.0, 179.75, LibraryUsage.USED)

    assert stac_item(used, PRODUCTION, PUBLICATION_DATETIME)["properties"]["visibility"] == ["default", "oceanbench"]
    assert stac_collection("public", (used,), PRODUCTION)["visibility"] == ["default", "oceanbench"]


def test_internal_catalog_is_private_and_has_its_own_identifiers():
    used = _bucket_dataset("used", "2024-01-03", "2024-12-25", -180.0, 179.75, LibraryUsage.USED)

    item = stac_item(used, INTERNAL, PUBLICATION_DATETIME)

    assert item["properties"]["visibility"] == ["oceanbench"]
    assert item["id"] == "oceanbench-used-internal"
    assert stac_catalog(INTERNAL)["id"] == "oceanbench-catalog-internal"
    assert stac_catalog(PRODUCTION)["id"] == "oceanbench-catalog"


def test_test_catalog_keeps_every_dataset_private():
    used = _bucket_dataset("used", "2024-01-03", "2024-12-25", -180.0, 179.75, LibraryUsage.USED)

    item = stac_item(used, CatalogEnvironment("ai-160"), PUBLICATION_DATETIME)

    assert item["properties"]["visibility"] == ["oceanbench"]


def test_public_item_hides_the_bucket_locations_and_storage_examples():
    used = _bucket_dataset("used", "2024-01-03", "2024-12-25", -180.0, 179.75, LibraryUsage.USED)

    item = stac_item(used, PRODUCTION, PUBLICATION_DATETIME)

    assert item["assets"] == {}
    assert not any(link["rel"] == "example" for link in item["links"])
    assert "oceanbench-bucket" not in json.dumps(item)
    assert "s3://" not in json.dumps(item)


def test_internal_item_keeps_the_bucket_locations_and_storage_examples_even_when_the_library_uses_it():
    used = _bucket_dataset("used", "2024-01-03", "2024-12-25", -180.0, 179.75, LibraryUsage.USED)

    item = stac_item(used, INTERNAL, PUBLICATION_DATETIME)

    assert item["assets"]["latest"]["href"].startswith("https://s3.waw3-1.cloudferro.com/oceanbench-bucket/")
    assert any(link["rel"] == "example" for link in item["links"])


def _example_titles(item: dict) -> list[str]:
    return [link["title"] for link in item["links"] if link["rel"] == "example"]


def test_internal_item_titles_tell_whether_the_library_uses_the_dataset_and_public_titles_do_not():
    used = _bucket_dataset("used", "2024-01-03", "2024-12-25", -180.0, 179.75, LibraryUsage.USED)
    unused = _bucket_dataset("unused", "2024-01-03", "2024-12-25", -180.0, 179.75)

    titles = tuple(
        stac_item(dataset, INTERNAL, PUBLICATION_DATETIME)["properties"]["title"] for dataset in (used, unused)
    )

    assert titles == ("public/used (used)", "public/unused (not used)")
    assert stac_item(used, PRODUCTION, PUBLICATION_DATETIME)["properties"]["title"] == "used"


def test_library_examples_come_from_the_functions_found_in_the_library_and_only_public_ones_are_shown_publicly():
    dataset = _bucket_dataset(
        "used",
        "2024-01-03",
        "2024-12-25",
        -180.0,
        179.75,
        LibraryUsage.USED,
        ("oceanbench.datasets.challenger.glonet", "oceanbench.core.ocean_mask.ocean_mask"),
    )

    assert _example_titles(stac_item(dataset, INTERNAL, PUBLICATION_DATETIME)) == [
        "Open with oceanbench.datasets.challenger.glonet",
        "Library function oceanbench.core.ocean_mask.ocean_mask",
        "Open the latest store with xarray",
    ]
    assert _example_titles(stac_item(dataset, PRODUCTION, PUBLICATION_DATETIME)) == [
        "Open with oceanbench.datasets.challenger.glonet"
    ]


def test_public_copernicus_marine_item_hides_the_store_url_and_keeps_the_toolbox_example():
    description = DatasetDescription(
        "glorys-twelfth-degree",
        "GLORYS",
        "copernicus-marine",
        CopernicusMarineDatasets(("cmems_mod_glo_phy_my_0.083deg_P1D-m",)),
        LibraryUsage.USED,
        (),
    )
    dataset = CopernicusMarineCatalogDataset(
        description,
        (
            CopernicusMarineDataset(
                "cmems_mod_glo_phy_my_0.083deg_P1D-m",
                "GLOBAL_MULTIYEAR_PHY_001_030",
                "Global Ocean Physics Reanalysis",
                "Description",
                "10.48670/moi-00021",
                "https://example.org/thumbnail.jpg",
                "https://example.org/arco/timeChunked.zarr",
                dataset_metadata(_weekly_forecast("2024-01-03", [0.0, 1.0], [-10.0, 10.0]), ()),
            ),
        ),
    )

    item = stac_item(dataset, PRODUCTION, PUBLICATION_DATETIME)

    assert list(item["assets"]) == ["thumbnail"]
    assert "arco" not in json.dumps(item)
    assert any(link["rel"] == "example" for link in item["links"])


def _write_library(directory: Path) -> Path:
    (directory / "core").mkdir(parents=True)
    (directory / "datasets").mkdir()
    (directory / "core" / "forecasts.py").write_text(
        'BUCKET_URL = "https://s3.example.org/oceanbench-bucket/public/forecasts"\n'
        "COPERNICUS_IDENTIFIER = 'cmems_mod_example_P1D-m'\n\n"
        "def _store_path(day):\n"
        '    return f"{BUCKET_URL}/{day}.zarr"\n\n'
        "def open_forecasts(day):\n"
        "    return _store_path(day), COPERNICUS_IDENTIFIER\n\n"
        "def mask():\n"
        '    return "https://s3.example.org/oceanbench-bucket/public/mask/mask.zarr"\n'
    )
    (directory / "datasets" / "challenger.py").write_text(
        "from oceanbench.core import forecasts\n\n" "def example(day):\n" "    return forecasts.open_forecasts(day)\n"
    )
    return directory


def test_library_usage_finds_bucket_paths_and_copernicus_marine_datasets_and_the_public_functions_leading_to_them(
    tmp_path,
):
    usage = library_usage(_write_library(tmp_path / "oceanbench"))

    assert [(item.path, item.public_functions) for item in usage.bucket] == [
        ("public/forecasts/", ("oceanbench.datasets.challenger.example",)),
        ("public/mask/mask.zarr", ("oceanbench.core.forecasts.mask",)),
    ]
    assert [(item.dataset_identifier, item.public_functions) for item in usage.copernicus_marine] == [
        ("cmems_mod_example_P1D-m", ("oceanbench.datasets.challenger.example",))
    ]


def test_bucket_directories_are_used_when_the_library_opens_stores_in_them_and_unused_otherwise(tmp_path):
    usage = library_usage(_write_library(tmp_path / "oceanbench"))

    descriptions = {
        description.identifier: description
        for description in dataset_descriptions(
            ("public/", "public/forecasts/", "public/mask/", "dev/forecasts/"), usage
        )
    }

    assert descriptions["public"].library_usage == LibraryUsage.UNUSED
    assert descriptions["public-forecasts"].library_usage == LibraryUsage.USED
    assert descriptions["public-mask"].library_usage == LibraryUsage.USED
    assert descriptions["dev-forecasts"].library_usage == LibraryUsage.UNUSED
    assert descriptions["cmems-mod-example-p1d-m"].collection == "copernicus-marine"


def test_the_bucket_datasets_used_by_the_real_library_are_found():
    paths = {item.path for item in library_usage().bucket}

    assert "public/ml-forecast-outputs/glonet/" in paths


def _health(bucket_object_count: BucketObjectCount | None) -> CatalogHealth:
    return CatalogHealth(PUBLICATION_DATETIME, 22, bucket_object_count, (), ())


def test_internal_health_holds_the_bucket_object_count_and_the_public_health_does_not():
    count = BucketObjectCount(94_779_357, datetime(2026, 11, 3, tzinfo=timezone.utc))

    assert health_metadata(_health(count))["bucket_object_count"] == {
        "count": 94_779_357,
        "counted": "2026-11-03T00:00:00+00:00",
    }
    assert "bucket_object_count" not in public_health_metadata(_health(count))


def test_bucket_object_count_is_read_back_from_the_stored_health():
    stored = health_metadata(_health(BucketObjectCount(5, PUBLICATION_DATETIME)))

    assert stored_bucket_object_count({"oceanbench:health": stored}) == BucketObjectCount(5, PUBLICATION_DATETIME)
    assert stored_bucket_object_count({"oceanbench:health": {"status": "healthy"}}) is None
    assert stored_bucket_object_count(None) is None


def test_bucket_object_count_is_reused_for_a_week_then_made_again():
    count = BucketObjectCount(5, PUBLICATION_DATETIME)

    assert reusable_bucket_object_count(count, PUBLICATION_DATETIME + timedelta(days=6, hours=23)) == count
    assert reusable_bucket_object_count(count, PUBLICATION_DATETIME + timedelta(days=7)) is None
    assert reusable_bucket_object_count(None, PUBLICATION_DATETIME) is None


def test_health_update_time_has_milliseconds_for_uptime_kuma():
    assert public_health_metadata(_health(None))["updated"] == "2026-10-30T00:00:00.000+00:00"
    assert health_metadata(_health(None))["updated"] == "2026-10-30T00:00:00.000+00:00"


class _FlakyRequests:
    def __init__(self, failures: tuple[Exception, ...]):
        self.failures = list(failures)
        self.calls = 0

    def request(self, method, url, **keyword_arguments):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return "response"


def _requests_failing_with(monkeypatch, *failures: Exception) -> _FlakyRequests:
    flaky = _FlakyRequests(failures)
    monkeypatch.setattr(edito_data_api.requests, "request", flaky.request)
    monkeypatch.setattr(edito_data_api.time, "sleep", lambda seconds: None)
    return flaky


def test_updates_are_retried_after_a_dropped_connection(monkeypatch):
    dropped = edito_data_api.requests.exceptions.ConnectionError("dropped")
    flaky = _requests_failing_with(monkeypatch, dropped, dropped)

    response = edito_data_api._request(edito_data_api.EditoSession("token"), "PUT", "https://example.org")

    assert (response, flaky.calls) == ("response", 3)


def test_creations_are_not_retried_when_the_request_may_have_been_received(monkeypatch):
    flaky = _requests_failing_with(monkeypatch, edito_data_api.requests.exceptions.ConnectionError("dropped"))

    with pytest.raises(edito_data_api.requests.exceptions.ConnectionError):
        edito_data_api._request(edito_data_api.EditoSession("token"), "POST", "https://example.org")

    assert flaky.calls == 1


def test_creations_are_retried_when_the_connection_was_never_established(monkeypatch):
    flaky = _requests_failing_with(monkeypatch, edito_data_api.requests.exceptions.SSLError("handshake"))

    edito_data_api._request(edito_data_api.EditoSession("token"), "POST", "https://example.org")

    assert flaky.calls == 2
