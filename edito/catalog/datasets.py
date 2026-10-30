# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass
from enum import Enum

VARIANT_GROUP = "variant"
DAILY_STORES = r"\d{8}\.zarr"
MONTHLY_STORES = r"\d{6}\.zarr"
EXPERIMENT_STORES = rf"(?P<{VARIANT_GROUP}>.+)/\d{{8}}\.zarr"


class CollectionKind(Enum):
    CHALLENGERS = "challengers"
    REFERENCES = "references"
    INPUTS = "inputs"
    TRAINING = "training"
    EVALUATIONS = "evaluations"


class LibraryUsage(Enum):
    USED = "used"
    UNUSED = "unused"


@dataclass(frozen=True)
class BucketObjects:
    prefix: str
    relative_key_pattern: str
    depth: int = 0


@dataclass(frozen=True)
class CopernicusMarineDatasets:
    dataset_identifiers: tuple[str, ...]


@dataclass(frozen=True)
class Period:
    first_day: str
    last_day: str


@dataclass(frozen=True)
class DatasetDescription:
    identifier: str
    title: str
    description: str
    collection: CollectionKind
    source: BucketObjects | CopernicusMarineDatasets
    library_usage: LibraryUsage
    library_function: str | None = None
    period: Period | None = None


GLORYS_TWELFTH_DEGREE = CopernicusMarineDatasets(("cmems_mod_glo_phy_my_0.083deg_P1D-m",))
GLO12_TWELFTH_DEGREE = CopernicusMarineDatasets(
    (
        "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
        "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
    )
)


def _challenger(
    identifier: str,
    title: str,
    description: str,
    source: BucketObjects,
    library_usage: LibraryUsage = LibraryUsage.UNUSED,
    library_function: str | None = None,
) -> DatasetDescription:
    return DatasetDescription(
        identifier, title, description, CollectionKind.CHALLENGERS, source, library_usage, library_function
    )


def _reference(
    identifier: str,
    title: str,
    description: str,
    source: BucketObjects | CopernicusMarineDatasets,
    library_usage: LibraryUsage = LibraryUsage.UNUSED,
    library_function: str | None = None,
) -> DatasetDescription:
    return DatasetDescription(
        identifier, title, description, CollectionKind.REFERENCES, source, library_usage, library_function
    )


def _input(
    identifier: str,
    title: str,
    description: str,
    source: BucketObjects,
    library_usage: LibraryUsage = LibraryUsage.UNUSED,
    library_function: str | None = None,
) -> DatasetDescription:
    return DatasetDescription(
        identifier, title, description, CollectionKind.INPUTS, source, library_usage, library_function
    )


def _evaluation(identifier: str, title: str, description: str, source: BucketObjects) -> DatasetDescription:
    return DatasetDescription(
        identifier, title, description, CollectionKind.EVALUATIONS, source, LibraryUsage.UNUSED, None
    )


CHALLENGER_DATASETS = (
    _challenger(
        "glonet",
        "GLONET forecasts",
        "Weekly 10-day global forecasts of the GLONET neural network (Mercator Ocean International), "
        "one Zarr store per forecast first day.",
        BucketObjects("public/ml-forecast-outputs/glonet/", DAILY_STORES),
        LibraryUsage.USED,
        "challenger.glonet",
    ),
    _challenger(
        "xihe",
        "XiHe forecasts",
        "Weekly 10-day global forecasts of the XiHe neural network, one Zarr store per forecast first day.",
        BucketObjects("public/ml-forecast-outputs/xihe/", DAILY_STORES),
        LibraryUsage.USED,
        "challenger.xihe",
    ),
    _challenger(
        "wenhai",
        "WenHai forecasts",
        "Weekly 10-day global forecasts of the WenHai neural network (version 2 outputs), "
        "one Zarr store per forecast first day.",
        BucketObjects("public/ml-forecast-outputs/wenhai/v2/", DAILY_STORES),
        LibraryUsage.USED,
        "challenger.wenhai",
    ),
    _challenger(
        "langya",
        "LangYa forecasts",
        "Weekly 7-day global forecasts of the LangYa neural network (version 2 outputs), "
        "one Zarr store per forecast first day.",
        BucketObjects("public/ml-forecast-outputs/langya/v2/", DAILY_STORES),
        LibraryUsage.USED,
        "challenger.langya",
    ),
    _challenger(
        "hclimrep",
        "HClimRep forecasts",
        "Weekly 10-day global forecasts of the HClimRep model, one Zarr store per forecast first day.",
        BucketObjects("public/ml-forecast-outputs/hclimrep/", DAILY_STORES),
        LibraryUsage.USED,
        "challenger.hclimrep",
    ),
    _challenger(
        "glo12-persistence",
        "GLO12 persistence baseline forecasts",
        "Persistence baseline: the GLO12 nowcast of the forecast first day repeated over every lead day.",
        BucketObjects("public/baseline-forecasts/persistence/", DAILY_STORES),
        LibraryUsage.USED,
        "challenger.glo12_persistence",
    ),
    _challenger(
        "glo36v1",
        "GLO36 version 1 forecasts",
        "Weekly 7-day global forecasts of the 1/36° GLO36 physical system, version 1, for 2023.",
        BucketObjects("public/glo36v1/", DAILY_STORES),
        LibraryUsage.USED,
        "challenger.glo36v1",
    ),
    _challenger(
        "glo12",
        "GLO12 forecasts",
        "Copernicus Marine GLO12 physical system forecasts, one Zarr store per run date with a group per variable.",
        BucketObjects("dev/additionnal-data/GLO12/", r"glo12_rg_1d-m_fcst_R\d{8}\.zarr"),
        LibraryUsage.USED,
        "challenger.glo12",
    ),
    _challenger(
        "glo12-climatology",
        "GLO12 climatology baseline forecasts",
        "Climatology baseline forecasts for 2024.",
        BucketObjects("public/baseline-forecasts/climatology/", DAILY_STORES),
    ),
    _challenger(
        "hclimrep-version-2",
        "HClimRep forecasts, version 2",
        "Second version of the HClimRep forecasts for 2024.",
        BucketObjects("public/ml-forecast-outputs/hclimrep/v2/", DAILY_STORES),
    ),
    _challenger(
        "langya-version-1",
        "LangYa forecasts, version 1",
        "First version of the LangYa forecasts, superseded by version 2.",
        BucketObjects("public/ml-forecast-outputs/langya/", DAILY_STORES),
    ),
    _challenger(
        "langya-quarantined",
        "LangYa forecasts, quarantined",
        "LangYa forecasts quarantined on 2026-06-23 because of a bare-load bug.",
        BucketObjects("public/_quarantine-langya-bareload-bug-20260623/", DAILY_STORES),
    ),
    _challenger(
        "glonet-full-2024",
        "GLONET forecasts, full 2024 legacy",
        "Legacy GLONET forecasts for 2024, superseded by the GLONET forecasts.",
        BucketObjects("public/glonet_full_2024/", DAILY_STORES),
    ),
    _challenger(
        "wenhai-2024-legacy",
        "WenHai forecasts, 2024 legacy",
        "Legacy WenHai forecasts for 2024, superseded by the WenHai forecasts.",
        BucketObjects("public/WENHAI/", DAILY_STORES),
    ),
    _challenger(
        "xihe-2024-legacy",
        "XiHe forecasts, 2024 legacy",
        "Legacy XiHe forecasts for 2024, superseded by the XiHe forecasts.",
        BucketObjects("public/XIHE/", DAILY_STORES),
    ),
    _challenger(
        "evaluation-input-{variant}",
        "Evaluation input forecasts {variant}",
        "Challenger forecasts staged as evaluation inputs, one Zarr store per forecast.",
        BucketObjects(
            "public/evaluation-inputs/challenger-forecasts/", rf"(?P<{VARIANT_GROUP}>[^/]+)/[^/]+\.zarr", depth=1
        ),
    ),
    _challenger(
        "development-{variant}",
        "Development forecasts {variant}",
        "Experimental forecasts under development, not part of the benchmark.",
        BucketObjects("dev/ml-forecast-outputs/", EXPERIMENT_STORES, depth=2),
    ),
    _challenger(
        "development-wenhai-divided-3600",
        "Development WenHai forecasts divided by 3600",
        "Experimental WenHai forecasts with a time unit correction, not part of the benchmark.",
        BucketObjects("dev/ml-forecast-outputs-wenhai-div3600/wenhai/", DAILY_STORES),
    ),
    _challenger(
        "development-smoke-{variant}",
        "Development smoke test forecasts {variant}",
        "Single forecast smoke tests of the inference pipelines.",
        BucketObjects("dev/ml-forecast-outputs-smoke/", rf"\d{{8}}/(?P<{VARIANT_GROUP}>[^/]+)/\d{{8}}\.zarr", depth=2),
    ),
)

REFERENCE_DATASETS = (
    _reference(
        "glorys-quarter-degree-2024",
        "GLORYS reanalysis, 1/4°, 2024",
        "GLORYS12 reanalysis interpolated on the 1/4° benchmark grid, one Zarr store per forecast first day.",
        BucketObjects("public/glorys14_refull_2024/", DAILY_STORES),
        LibraryUsage.USED,
    ),
    _reference(
        "glorys-one-degree-2024",
        "GLORYS reanalysis, 1°, 2024",
        "GLORYS12 reanalysis interpolated on the 1° benchmark grid, one Zarr store per forecast first day.",
        BucketObjects("public/glorys_1degree_2024_V2/", DAILY_STORES),
        LibraryUsage.USED,
    ),
    _reference(
        "glorys-twelfth-degree",
        "GLORYS reanalysis, 1/12°",
        "GLORYS12 daily reanalysis at native 1/12° resolution, served by Copernicus Marine.",
        GLORYS_TWELFTH_DEGREE,
        LibraryUsage.USED,
        "reference.glorys_reanalysis",
    ),
    _reference(
        "glo12-analysis-quarter-degree-2024",
        "GLO12 analysis, 1/4°, 2024",
        "GLO12 analysis interpolated on the 1/4° benchmark grid, one Zarr store per forecast first day.",
        BucketObjects("public/glo14/", DAILY_STORES),
        LibraryUsage.USED,
    ),
    _reference(
        "glo12-analysis-one-degree-2024",
        "GLO12 analysis, 1°, 2024",
        "GLO12 analysis interpolated on the 1° benchmark grid, one Zarr store per forecast first day.",
        BucketObjects("public/glo12_1degree_2024_V2/", DAILY_STORES),
        LibraryUsage.USED,
    ),
    _reference(
        "glo12-analysis-twelfth-degree",
        "GLO12 analysis, 1/12°",
        "GLO12 daily analysis at native 1/12° resolution, served by Copernicus Marine.",
        GLO12_TWELFTH_DEGREE,
        LibraryUsage.USED,
        "reference.glo12_analysis",
    ),
    _reference(
        "observations-2024-version-2",
        "Observations 2024, version 2",
        "Daily in situ and satellite observations (Argo, drifters, moorings, SST, SLA) used for Class 4 "
        "evaluation, one Zarr store and manifest per day.",
        BucketObjects("public/observations2024-v2/", DAILY_STORES),
        LibraryUsage.USED,
    ),
    _reference(
        "mean-dynamic-topography-glo12",
        "GLO12 mean dynamic topography",
        "GLO12 mean dynamic topography at 1/12°, 1/4° and 1°, used to compare sea level anomalies.",
        BucketObjects("public/GLO12_MDT/", r"GLO-MFC_001_024_mdt.*\.zarr"),
        LibraryUsage.USED,
    ),
    _reference(
        "ocean-mask-version-2",
        "OceanBench ocean mask, version 2",
        "Ocean mask combining the GLO12 and GLORYS static masks from Copernicus Marine "
        "(cmems_mod_glo_phy_anfc_0.083deg_static, cmems_mod_glo_phy_my_0.083deg_static).",
        BucketObjects("public/ocean_mask/", r"oceanbench-ocean-mask-v2\.zarr"),
        LibraryUsage.USED,
    ),
    _reference(
        "reference-{variant}",
        "Reference {variant}",
        "Reference analyses and reanalyses for other evaluation years, one Zarr store per forecast first day.",
        BucketObjects("public/references/", rf"(?P<{VARIANT_GROUP}>[^/]+)/\d{{8}}\.zarr", depth=1),
    ),
    _reference(
        "marine-heatwave-climatology",
        "Marine heatwave climatology",
        "GLORYS12 1993-2022 daily climatologies used to detect marine heatwaves.",
        BucketObjects("public/mhw/climatology/", r".*\.zarr"),
    ),
    _reference(
        "raw-observation-archive",
        "Raw observation archive",
        "Archive of raw Copernicus Marine in situ and sea level observations, stored as "
        "`<product>/<dataset>/<YYYYMMDD>/<file>.nc` with a JSON metadata file per NetCDF file. "
        "The objects of this item are the archive index files.",
        BucketObjects("dev/obs-raw-archive/", r"index/[^/]+\.jsonl", depth=1),
    ),
    _reference(
        "glo12-analysis-2024-legacy",
        "GLO12 analysis, 2024 legacy",
        "Legacy GLO12 analysis for 2024, superseded by the 1/4° GLO12 analysis.",
        BucketObjects("public/GLO12/", DAILY_STORES),
    ),
    _reference(
        "glo12-analysis-quarter-degree-2024-uppercase",
        "GLO12 analysis, 1/4°, 2024, uppercase prefix",
        "Duplicate of the 1/4° GLO12 analysis for 2024 under an uppercase prefix.",
        BucketObjects("public/GLO14/", DAILY_STORES),
    ),
    _reference(
        "glo12-analysis-quarter-degree-2024-netcdf",
        "GLO12 analysis, 1/4°, 2024, NetCDF",
        "NetCDF version of the 1/4° GLO12 analysis for 2024.",
        BucketObjects("public/GLO14/", r"\d{8}\.nc"),
    ),
    _reference(
        "mean-dynamic-topography-glo12-netcdf",
        "GLO12 mean dynamic topography, NetCDF",
        "NetCDF version of the GLO12 mean dynamic topography.",
        BucketObjects("public/GLO12_MDT/", r".*\.nc"),
    ),
    _reference(
        "mean-dynamic-topography-glorys-2024",
        "GLORYS mean dynamic topography, 2024",
        "GLORYS12 mean dynamic topography at several resolutions.",
        BucketObjects("public/glorys12_mdt_2024/", r".*\.zarr"),
    ),
    _reference(
        "mean-sea-surface-height-glorys-2024",
        "GLORYS mean sea surface height, 2024",
        "GLORYS12 mean sea surface height at several resolutions.",
        BucketObjects("public/glorys12_mean_sea_surface_height_2024/", r".*\.zarr"),
    ),
    _reference(
        "mean-sea-surface-height-glorys-2024-single-store",
        "GLORYS mean sea surface height, 2024, single store",
        "GLORYS12 mean sea surface height in a single store at the root of the public prefix.",
        BucketObjects("public/", r"glorys12_mean_sea_surface_height_2024\.zarr"),
    ),
    _reference(
        "class4-mean-sea-surface-height-glorys-2024",
        "GLORYS mean sea surface height for Class 4, 2024",
        "GLORYS12 mean sea surface height used by the legacy Class 4 evaluation.",
        BucketObjects("class4/Glorys_mssh2024/", r".*\.zarr"),
    ),
    _reference(
        "class4-statics",
        "Class 4 static fields",
        "Mean dynamic topography NetCDF files used by the legacy Class 4 evaluation.",
        BucketObjects("class4/statics/", r".*\.nc"),
    ),
    _reference(
        "glo12-analysis-one-degree-2024-version-1",
        "GLO12 analysis, 1°, 2024, version 1",
        "First version of the 1° GLO12 analysis, superseded by version 2.",
        BucketObjects("public/glo12_1degree_2024/", DAILY_STORES),
    ),
    _reference(
        "glorys-one-degree-2024-version-1",
        "GLORYS reanalysis, 1°, 2024, version 1",
        "First version of the 1° GLORYS reanalysis, superseded by version 2.",
        BucketObjects("public/glorys_1degree_2024/", DAILY_STORES),
    ),
    _reference(
        "glo12-analysis-one-degree-2024-coarsened",
        "GLO12 analysis, 1°, 2024, coarsened",
        "1° GLO12 analysis obtained by coarsening instead of interpolation.",
        BucketObjects("public/glo12_1degree_2024_coarsened/", DAILY_STORES),
    ),
    _reference(
        "glorys-one-degree-2024-coarsened",
        "GLORYS reanalysis, 1°, 2024, coarsened",
        "1° GLORYS reanalysis obtained by coarsening instead of interpolation.",
        BucketObjects("public/glorys_1degree_2024_coarsened/", DAILY_STORES),
    ),
    _reference(
        "glorys-quarter-degree-2024-legacy",
        "GLORYS reanalysis, 1/4°, 2024 legacy",
        "Legacy 1/4° GLORYS reanalysis for 2024, superseded by the 1/4° GLORYS reanalysis.",
        BucketObjects("public/glorys14_full_2024/", DAILY_STORES),
    ),
    _reference(
        "ibirys36-on-glorys-grid",
        "IBIRYS36 reanalysis on the GLORYS grid",
        "Iberia-Biscay-Ireland 1/36° reanalysis interpolated on the 1/12° GLORYS grid, one Zarr store per month.",
        BucketObjects("public/ibirys36_1-12_glorys-grid/", MONTHLY_STORES),
    ),
    _reference(
        "observations-2023",
        "Observations 2023",
        "Daily observations for 2023, matching the GLO36 version 1 forecasts.",
        BucketObjects("public/observations2023/", DAILY_STORES),
    ),
    _reference(
        "observations-2024-version-1",
        "Observations 2024, version 1",
        "First version of the 2024 observations, superseded by version 2.",
        BucketObjects("public/observations2024/", DAILY_STORES),
    ),
    _reference(
        "observations-2025",
        "Observations 2025",
        "Daily observations from 2025 onwards.",
        BucketObjects("public/observations2025/", DAILY_STORES),
    ),
    _reference(
        "observations-final-version-2",
        "Observations, final version 2 sample",
        "Sample of 15 days of observations from January 2024.",
        BucketObjects("public/obs_finales_v2/", DAILY_STORES),
    ),
    _reference(
        "live-observations",
        "Live observations",
        "Rolling window of the latest daily observations, maintained by the daily-observation-data EDITO process.",
        BucketObjects("public/live_observations/", DAILY_STORES),
    ),
    _reference(
        "development-observations-2024-version-2",
        "Development observations 2024, version 2",
        "Development copy of the 2024 observations, version 2.",
        BucketObjects("dev/observations2024-v2/", DAILY_STORES),
    ),
    _reference(
        "development-observations-2025-version-2",
        "Development observations 2025, version 2",
        "2025 observations, version 2, before promotion to the public prefix.",
        BucketObjects("dev/observations2025-v2/", DAILY_STORES),
    ),
    _reference(
        "development-observations-2025-version-2-smoke",
        "Development observations 2025, version 2, smoke test",
        "Three days of 2025 observations, version 2, to test the observation pipeline.",
        BucketObjects("dev/observations2025-v2-smoke/", DAILY_STORES),
    ),
    _reference(
        "development-ocean-masks",
        "Development ocean masks",
        "Development copies of the OceanBench ocean masks.",
        BucketObjects("dev/ocean_mask/", r"oceanbench-ocean-mask-v\d+\.zarr"),
    ),
    _reference(
        "swot-level-3-{variant}-netcdf",
        "SWOT level 3 sea surface height, {variant}, NetCDF",
        "SWOT level 3 low resolution basic sea surface height passes, NetCDF files stored by year.",
        BucketObjects("class4/swot/l3/", rf"(?P<{VARIANT_GROUP}>forward|reproc)/\d{{4}}/SWOT_.*\.nc", depth=2),
    ),
    _reference(
        "swot-level-3-{variant}",
        "SWOT level 3 sea surface height, {variant}",
        "SWOT level 3 low resolution basic sea surface height, daily Zarr stores stored by year.",
        BucketObjects("class4/swot/l3/zarr/", rf"(?P<{VARIANT_GROUP}>forward|reproc)/\d{{4}}/\d{{8}}\.zarr", depth=2),
    ),
    _reference(
        "class4-filtered-currents-2024-archive",
        "Filtered currents 2024 archive",
        "Archive of 2024 filtered current observations used by the legacy Class 4 evaluation.",
        BucketObjects("class4/", r"013_048_filtr_2024\.tar"),
    ),
)

INPUT_DATASETS = (
    _input(
        "glo12-nowcasts",
        "GLO12 nowcasts",
        "GLO12 nowcasts used as initial states by challengers, one Zarr store per run date with a group per variable.",
        BucketObjects("dev/additionnal-data/GLO12/", r"glo12_rg_1d-m_nwct_R\d{8}\.zarr"),
        LibraryUsage.USED,
        "input.glo12_nowcasts",
    ),
    _input(
        "ifs-forcings",
        "IFS forcings",
        "ECMWF IFS atmospheric forecasts used as forcings by challengers, one Zarr store per run date.",
        BucketObjects("dev/additionnal-data/IFS/", r"ifs_forcing_rg_forecasts_R\d{8}\.zarr"),
        LibraryUsage.USED,
        "input.ifs_forcings",
    ),
    _input(
        "ifs-forcings-2024-netcdf",
        "IFS forcings 2024, NetCDF",
        "ECMWF IFS atmospheric forcings for 2024 described in the OceanBench input datasets Croissant metadata.",
        BucketObjects("public/IFS/", r"IFS_\d{8}\.nc"),
    ),
    _input(
        "glo12-nowcasts-2024",
        "GLO12 nowcasts 2024",
        "GLO12 nowcasts for 2024 used to generate the persistence baseline.",
        BucketObjects("public/GLO12_NOWCAST/", DAILY_STORES),
    ),
    _input(
        "glo12-nowcasts-native-grid",
        "GLO12 nowcasts on the native grid",
        "GLO12 nowcasts on the native ORCA12 grid, NetCDF files per variable and day.",
        BucketObjects("public/GLO12_NOWCAST_NATIVE/", r".*\.nc"),
    ),
    _input(
        "ifs-nowcasts",
        "IFS nowcasts",
        "ECMWF IFS atmospheric nowcasts, one Zarr store per day.",
        BucketObjects("public/ifs-nowcast/", DAILY_STORES),
    ),
    _input(
        "ifs-nowcasts-corrected",
        "IFS nowcasts, corrected",
        "Corrected ECMWF IFS atmospheric nowcasts, one Zarr store per day.",
        BucketObjects("public/ifs-nowcast-corrected/", DAILY_STORES),
    ),
    _input(
        "ifs-nowcasts-24-hours",
        "IFS nowcasts, 24 hours",
        "ECMWF IFS atmospheric 24-hour nowcasts, one Zarr store per day.",
        BucketObjects("public/ifs-nowcasts24/", DAILY_STORES),
    ),
    _input(
        "ifs-forcings-legacy",
        "IFS forcings, legacy names",
        "ECMWF IFS forcings stored under legacy names.",
        BucketObjects("dev/additionnal-data/IFS/", r"ocean_bench_R\d{8}\.zarr"),
    ),
)

TRAINING_DATASETS = (
    DatasetDescription(
        "glorys-one-degree-1993-2019",
        "GLORYS reanalysis, 1°, 1993-2019",
        "GLORYS12 reanalysis interpolated on the 1° grid for training, one Zarr store per month.",
        CollectionKind.TRAINING,
        BucketObjects("public/glorys_1degree_1993_2019/", MONTHLY_STORES),
        LibraryUsage.USED,
        "training.glorys_reanalysis_1_degree",
    ),
    DatasetDescription(
        "glorys-one-degree-1993-2019-reference",
        "GLORYS reanalysis, 1°, 1993-2019, reference",
        "GLORYS12 reanalysis interpolated on the 1° grid, exposed as a reference, one Zarr store per month.",
        CollectionKind.REFERENCES,
        BucketObjects("public/glorys_1degree_1993_2019/", MONTHLY_STORES),
        LibraryUsage.USED,
        "reference.glorys_reanalysis_1_degree",
    ),
    DatasetDescription(
        "glorys-twelfth-degree-1993-2019",
        "GLORYS reanalysis, 1/12°, 1993-2019",
        "GLORYS12 daily reanalysis at native 1/12° resolution for training, served by Copernicus Marine.",
        CollectionKind.TRAINING,
        GLORYS_TWELFTH_DEGREE,
        LibraryUsage.USED,
        "training.glorys_reanalysis",
        Period("1993-01-01", "2019-12-31"),
    ),
)

EVALUATION_DATASETS = (
    _evaluation(
        "class4-{variant}",
        "Class 4 matchups {variant}",
        "Daily Class 4 files matching a system with observations, from the legacy Class 4 evaluation.",
        BucketObjects("class4/", rf"class4_\d{{8}}_(?P<{VARIANT_GROUP}>.+)\.nc"),
    ),
    _evaluation(
        "class4-sea-level-anomaly-foam-2018",
        "Class 4 sea level anomaly matchups of FOAM, 2018",
        "Daily Class 4 sea level anomaly files of the Met Office FOAM system for 2018.",
        BucketObjects("class4/SLA/", r".*\.nc"),
    ),
    _evaluation(
        "class4-scores",
        "Class 4 scores",
        "Class 4 scores per system for 2024.",
        BucketObjects("class4/", r"score.*\.csv"),
    ),
    _evaluation(
        "class4-scores-per-system",
        "Class 4 scores per system",
        "Class 4 score NetCDF files per system for 2024.",
        BucketObjects("class4/scores/", r".*\.nc"),
    ),
    _evaluation(
        "scores-{variant}",
        "Scores against {variant}",
        "Legacy score NetCDF files per challenger.",
        BucketObjects("scores/", rf"(?P<{VARIANT_GROUP}>[^/]+)/[^/]+\.nc", depth=1),
    ),
    _evaluation(
        "ola-sea-level-anomaly-ibirys36-2024",
        "IBIRYS36 sea level anomaly observation-minus-analysis, 2024",
        "IBIRYS36 sea level anomaly observation-minus-analysis catalogs for 2024.",
        BucketObjects("dev/additionnal-data/OLA/", r".*\.dat"),
    ),
    _evaluation(
        "ensemble-sigma-lookup",
        "Ensemble sigma lookup tables",
        "Lookup tables of ensemble spread used by the ensemble evaluation.",
        BucketObjects("dev/ensemble/", r"sigma-lookup-v[\d.]+\.zarr"),
    ),
    _evaluation(
        "ensemble-sigma-lookup-artifact",
        "Ensemble sigma lookup table artifact",
        "Archived lookup table of ensemble spread.",
        BucketObjects("dev/artifacts/", r"sigma-lookup-v[\d.]+\.zarr"),
    ),
    _evaluation(
        "ensemble-scores-{variant}",
        "Ensemble scores {variant}",
        "Class 4 and gridded scores and rank histograms of an ensemble experiment.",
        BucketObjects("dev/ensemble/", rf"(?P<{VARIANT_GROUP}>[^/]+)/(class4|gridded)/[^/]+\.parquet", depth=2),
    ),
)

NOT_DATASET_PREFIXES = (
    "public/evaluation-reports/",
    "public/evaluation-reports-multi-year/",
    "public/logo/",
    "public/ifs-nowcasts/",
    "dev/evaluation-reports/",
    "dev/benchmark/",
    "dev/ml-forecast-assets/",
)

DATASET_DESCRIPTIONS = (
    *CHALLENGER_DATASETS,
    *REFERENCE_DATASETS,
    *INPUT_DATASETS,
    *TRAINING_DATASETS,
    *EVALUATION_DATASETS,
)
