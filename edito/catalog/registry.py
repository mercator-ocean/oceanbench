# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass
from datetime import date
from enum import Enum

BUCKET_URL = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket"
CHALLENGER_FIRST_DAY = date(2024, 1, 3)
CHALLENGER_LAST_DAY = date(2025, 1, 3)
REFERENCE_FIRST_DAY = date(2024, 1, 2)
REFERENCE_LAST_DAY = date(2024, 12, 31)
BENCHMARK_FIRST_DAY = date(2024, 1, 1)
TRAINING_FIRST_DAY = date(1993, 1, 1)
TRAINING_LAST_DAY = date(2019, 12, 31)
GLORYS_REANALYSIS_DATASET_IDENTIFIER = "cmems_mod_glo_phy_my_0.083deg_P1D-m"


class OneDegreeVariant(Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class Collection(Enum):
    INPUTS = "inputs"
    CHALLENGERS = "challengers"
    REFERENCES = "references"
    TRAINING = "training"
    AUXILIARY = "auxiliary"


@dataclass(frozen=True)
class TemporalExtent:
    first_day: date
    last_day: date


@dataclass(frozen=True)
class BucketStores:
    sample_path: str
    path_pattern: str


@dataclass(frozen=True)
class CopernicusMarineDatasets:
    product_identifier: str
    dataset_identifiers: tuple[str, ...]


@dataclass(frozen=True)
class CatalogEntry:
    identifier: str
    collection: Collection
    title: str
    description: str
    library_functions: tuple[str, ...]
    source: BucketStores | CopernicusMarineDatasets
    temporal_extent: TemporalExtent | None


def _challenger(
    identifier: str,
    title: str,
    path_pattern: str,
    one_degree_variant: OneDegreeVariant = OneDegreeVariant.AVAILABLE,
    sample_first_day: str = "20240103",
) -> CatalogEntry:
    return CatalogEntry(
        identifier=f"challenger-{identifier.replace('_', '-')}",
        collection=Collection.CHALLENGERS,
        title=title,
        description=f"{title} weekly forecasts of 10 lead days, one store per first forecast day.",
        library_functions=(
            f"oceanbench.datasets.challenger.{identifier}",
            *(
                (f"oceanbench.datasets.challenger.{identifier}_1_degree",)
                if one_degree_variant == OneDegreeVariant.AVAILABLE
                else ()
            ),
        ),
        source=BucketStores(path_pattern.format(first_day=sample_first_day), path_pattern),
        temporal_extent=TemporalExtent(CHALLENGER_FIRST_DAY, CHALLENGER_LAST_DAY),
    )


CATALOG_ENTRIES = (
    CatalogEntry(
        identifier="input-glo12-nowcasts",
        collection=Collection.INPUTS,
        title="GLO12 nowcasts",
        description="GLO12 nowcasts used as initial conditions of the challengers, one store per week.",
        library_functions=("oceanbench.datasets.input.glo12_nowcasts",),
        source=BucketStores(
            "dev/additionnal-data/GLO12/glo12_rg_1d-m_nwct_R20240103.zarr",
            "dev/additionnal-data/GLO12/glo12_rg_1d-m_nwct_R{first_day}.zarr",
        ),
        temporal_extent=TemporalExtent(date(2022, 12, 28), date(2025, 12, 30)),
    ),
    CatalogEntry(
        identifier="input-ifs-forcings",
        collection=Collection.INPUTS,
        title="IFS atmospheric forcings",
        description="ECMWF IFS atmospheric forecasts used to force the challengers, one store per week.",
        library_functions=("oceanbench.datasets.input.ifs_forcings",),
        source=BucketStores(
            "dev/additionnal-data/IFS/ifs_forcing_rg_forecasts_R20230103.zarr",
            "dev/additionnal-data/IFS/ifs_forcing_rg_forecasts_R{first_day}.zarr",
        ),
        temporal_extent=TemporalExtent(date(2023, 1, 3), date(2025, 12, 31)),
    ),
    _challenger(
        "glo12",
        "GLO12 forecasts",
        "dev/additionnal-data/GLO12/glo12_rg_1d-m_fcst_R{first_day}.zarr",
        sample_first_day="20240104",
    ),
    _challenger("glo36v1", "GLO36 v1 forecasts", "public/glo36v1/{first_day}.zarr", OneDegreeVariant.UNAVAILABLE),
    _challenger("glonet", "GLONET forecasts", "public/ml-forecast-outputs/glonet/{first_day}.zarr"),
    _challenger("xihe", "XiHe forecasts", "public/ml-forecast-outputs/xihe/{first_day}.zarr"),
    _challenger("wenhai", "WenHai forecasts", "public/ml-forecast-outputs/wenhai/v2/{first_day}.zarr"),
    _challenger("langya", "LangYa forecasts", "public/ml-forecast-outputs/langya/v2/{first_day}.zarr"),
    _challenger("hclimrep", "HClimRep forecasts", "public/ml-forecast-outputs/hclimrep/{first_day}.zarr"),
    _challenger(
        "glo12_persistence",
        "GLO12 persistence forecasts",
        "public/baseline-forecasts/persistence/{first_day}.zarr",
    ),
    _challenger("climatology", "GLORYS climatology", "public/baseline-forecasts/climatology/{first_day}.zarr"),
    CatalogEntry(
        identifier="challenger-glorys",
        collection=Collection.CHALLENGERS,
        title="GLORYS reanalysis as a challenger",
        description="GLORYS reanalysis from Copernicus Marine scored as a challenger, its lead days the valid dates.",
        library_functions=("oceanbench.datasets.challenger.glorys",),
        source=CopernicusMarineDatasets("GLOBAL_MULTIYEAR_PHY_001_030", (GLORYS_REANALYSIS_DATASET_IDENTIFIER,)),
        temporal_extent=TemporalExtent(BENCHMARK_FIRST_DAY, REFERENCE_LAST_DAY),
    ),
    _challenger(
        "glorys_1_degree",
        "GLORYS reanalysis at 1° as a challenger",
        "public/glorys_1degree_2024_V2/{first_day}.zarr",
        OneDegreeVariant.UNAVAILABLE,
    ),
    CatalogEntry(
        identifier="challenger-glorys-twelfth-degree",
        collection=Collection.CHALLENGERS,
        title="GLORYS reanalysis at 1/12° as a challenger",
        description="GLORYS 1/12° reanalysis from Copernicus Marine scored as a challenger, its lead days the valid dates.",
        library_functions=("oceanbench.datasets.challenger.glorys_twelfth_degree",),
        source=CopernicusMarineDatasets("GLOBAL_MULTIYEAR_PHY_001_030", (GLORYS_REANALYSIS_DATASET_IDENTIFIER,)),
        temporal_extent=TemporalExtent(BENCHMARK_FIRST_DAY, REFERENCE_LAST_DAY),
    ),
    CatalogEntry(
        identifier="challenger-glo12-analysis",
        collection=Collection.CHALLENGERS,
        title="GLO12 analysis as a challenger",
        description="GLO12 1/12° analysis from Copernicus Marine scored as a challenger, its lead days the valid dates.",
        library_functions=("oceanbench.datasets.challenger.glo12_analysis",),
        source=CopernicusMarineDatasets(
            "GLOBAL_ANALYSISFORECAST_PHY_001_024",
            (
                "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
                "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
                "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
                "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            ),
        ),
        temporal_extent=TemporalExtent(BENCHMARK_FIRST_DAY, REFERENCE_LAST_DAY),
    ),
    CatalogEntry(
        identifier="reference-glorys-reanalysis",
        collection=Collection.REFERENCES,
        title="GLORYS reanalysis",
        description="GLORYS 1/12° reanalysis from Copernicus Marine used as reference.",
        library_functions=("oceanbench.datasets.reference.glorys_reanalysis",),
        source=CopernicusMarineDatasets("GLOBAL_MULTIYEAR_PHY_001_030", (GLORYS_REANALYSIS_DATASET_IDENTIFIER,)),
        temporal_extent=TemporalExtent(BENCHMARK_FIRST_DAY, REFERENCE_LAST_DAY),
    ),
    CatalogEntry(
        identifier="reference-glorys-reanalysis-1-degree",
        collection=Collection.REFERENCES,
        title="GLORYS reanalysis at 1°",
        description="GLORYS reanalysis interpolated at 1° used as reference, one store per day.",
        library_functions=("oceanbench.datasets.reference.glorys_reanalysis_1_degree",),
        source=BucketStores("public/glorys_1degree_2024_V2/20240103.zarr", "public/glorys_1degree_2024_V2/{day}.zarr"),
        temporal_extent=TemporalExtent(REFERENCE_FIRST_DAY, REFERENCE_LAST_DAY),
    ),
    CatalogEntry(
        identifier="reference-glo12-analysis",
        collection=Collection.REFERENCES,
        title="GLO12 analysis",
        description="GLO12 1/12° analysis from Copernicus Marine used as reference.",
        library_functions=("oceanbench.datasets.reference.glo12_analysis",),
        source=CopernicusMarineDatasets(
            "GLOBAL_ANALYSISFORECAST_PHY_001_024",
            (
                "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
                "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
                "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
                "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            ),
        ),
        temporal_extent=TemporalExtent(BENCHMARK_FIRST_DAY, REFERENCE_LAST_DAY),
    ),
    CatalogEntry(
        identifier="reference-glo12-analysis-1-degree",
        collection=Collection.REFERENCES,
        title="GLO12 analysis at 1°",
        description="GLO12 analysis interpolated at 1° used as reference for 1° challengers, one store per day.",
        library_functions=(),
        source=BucketStores("public/glo12_1degree_2024_V2/20240103.zarr", "public/glo12_1degree_2024_V2/{day}.zarr"),
        temporal_extent=TemporalExtent(REFERENCE_FIRST_DAY, REFERENCE_LAST_DAY),
    ),
    CatalogEntry(
        identifier="reference-observations",
        collection=Collection.REFERENCES,
        title="Observations",
        description="In situ and satellite observations used as reference, one store per day.",
        library_functions=(),
        source=BucketStores("public/observations2024-v2/20240103.zarr", "public/observations2024-v2/{day}.zarr"),
        temporal_extent=TemporalExtent(REFERENCE_FIRST_DAY, REFERENCE_LAST_DAY),
    ),
    CatalogEntry(
        identifier="training-glorys-reanalysis",
        collection=Collection.TRAINING,
        title="GLORYS reanalysis",
        description="GLORYS 1/12° reanalysis from Copernicus Marine for training.",
        library_functions=("oceanbench.datasets.training.glorys_reanalysis",),
        source=CopernicusMarineDatasets(
            "GLOBAL_MULTIYEAR_PHY_001_030",
            (GLORYS_REANALYSIS_DATASET_IDENTIFIER, "cmems_mod_glo_phy_my_0.083deg_static"),
        ),
        temporal_extent=TemporalExtent(date(1993, 1, 2), TRAINING_LAST_DAY),
    ),
    CatalogEntry(
        identifier="training-glorys-reanalysis-1-degree",
        collection=Collection.TRAINING,
        title="GLORYS reanalysis at 1°",
        description="GLORYS reanalysis interpolated at 1° for training, one store per month.",
        library_functions=("oceanbench.datasets.training.glorys_reanalysis_1_degree",),
        source=BucketStores(
            "public/glorys_1degree_1993_2019/199301.zarr", "public/glorys_1degree_1993_2019/{month}.zarr"
        ),
        temporal_extent=TemporalExtent(TRAINING_FIRST_DAY, TRAINING_LAST_DAY),
    ),
    CatalogEntry(
        identifier="auxiliary-ocean-mask",
        collection=Collection.AUXILIARY,
        title="Ocean mask",
        description="Ocean mask applied to every dataset before evaluation.",
        library_functions=(),
        source=BucketStores(
            "public/ocean_mask/oceanbench-ocean-mask-v2.zarr", "public/ocean_mask/oceanbench-ocean-mask-v2.zarr"
        ),
        temporal_extent=None,
    ),
    CatalogEntry(
        identifier="auxiliary-mean-dynamic-topography",
        collection=Collection.AUXILIARY,
        title="Mean dynamic topography",
        description="GLO12 mean dynamic topography at 1/12°, 1/4° and 1°, used to compare sea surface heights.",
        library_functions=(),
        source=BucketStores(
            "public/GLO12_MDT/GLO-MFC_001_024_mdt.zarr", "public/GLO12_MDT/GLO-MFC_001_024_mdt{resolution}.zarr"
        ),
        temporal_extent=None,
    ),
)


def bucket_url(path: str) -> str:
    return f"{BUCKET_URL}/{path}"
