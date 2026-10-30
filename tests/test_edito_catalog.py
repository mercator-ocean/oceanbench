# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy
import pandas
import xarray

CATALOG_DIRECTORY = Path(__file__).resolve().parents[1] / "edito" / "catalog"
sys.path.insert(0, str(CATALOG_DIRECTORY))

from bucket import BucketObject, ObjectKind  # noqa: E402
from dataset_metadata import dataset_metadata, merged_metadata  # noqa: E402
from dataset_resolution import BucketDataset  # noqa: E402
from datasets import BucketObjects, CollectionKind, DatasetDescription, LibraryUsage  # noqa: E402
from stac import CatalogEnvironment, stac_collection, stac_item  # noqa: E402

LEAD_DAYS_COUNT = 7
PRODUCTION = CatalogEnvironment(None)
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


def _bucket_dataset(identifier: str, first_day: str, last_day: str, west: float, east: float) -> BucketDataset:
    first_store = _weekly_forecast(first_day, [west, 0.0], [-10.0, 10.0])
    last_store = _weekly_forecast(last_day, [0.0, east], [-10.0, 10.0])
    return BucketDataset(
        description=DatasetDescription(
            identifier,
            identifier,
            "Forecasts",
            CollectionKind.CHALLENGERS,
            BucketObjects(f"public/{identifier}/", r"\d{8}\.zarr"),
            LibraryUsage.UNUSED,
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
        CollectionKind.CHALLENGERS,
        (
            _bucket_dataset("forecasts-2023", "2023-01-04", "2023-12-27", -180.0, 0.0),
            _bucket_dataset("forecasts-2024", "2024-01-03", "2024-12-25", -90.0, 179.75),
        ),
        PRODUCTION,
    )

    assert collection["extent"]["spatial"]["bbox"] == [[-180.0, -10.0, 179.75, 10.0]]
    assert collection["extent"]["temporal"]["interval"] == [["2023-01-04T00:00:00+00:00", "2024-12-31T00:00:00+00:00"]]
