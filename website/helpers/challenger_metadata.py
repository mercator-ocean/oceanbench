# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

_GLO12 = {
    "label": "GLO12",
    "url": "https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024",
    "organisation": "Mercator Ocean",
    "organisation_url": "https://mercator-ocean.eu",
    "method": "Physics-based",
    "forecast_type": "Deterministic",
    "initial_conditions": "IFS HRES",
    "resolution": "1/12°",
}

_GLONET = {
    "label": "GLONET",
    "url": "https://glonet.lab.dive.edito.eu",
    "organisation": "Mercator Ocean",
    "organisation_url": "https://mercator-ocean.eu",
    "method": "ML-based",
    "forecast_type": "Deterministic",
    "initial_conditions": "GLO12",
    "resolution": "1/4°",
}

_WENHAI = {
    "label": "WenHai",
    "url": "https://www.nature.com/articles/s41467-025-57389-2",
    "organisation": "DOMES",
    "organisation_url": "http://iaos.ouc.edu.cn/DeepOceanMultispheresandEarthSystem/list.htm",
    "method": "ML-based",
    "forecast_type": "Deterministic",
    "initial_conditions": "GLO12/IFS",
    "resolution": "1/12°",
}

_XIHE = {
    "label": "XiHe",
    "url": "https://arxiv.org/abs/2402.02995",
    "organisation": "NUDT",
    "organisation_url": "https://english.nudt.edu.cn/",
    "method": "ML-based",
    "forecast_type": "Deterministic",
    "initial_conditions": "GLO12/IFS",
    "resolution": "1/12°",
}

_LANGYA = {
    "label": "LangYa",
    "url": "https://arxiv.org/abs/2412.18097",
    "organisation": "IOCAS",
    "organisation_url": "http://english.qdio.cas.cn/",
    "method": "ML-based",
    "forecast_type": "Deterministic",
    "initial_conditions": "GLO12/IFS",
    "resolution": "1/12°",
}

_GLO12_PERSISTENCE = {
    "label": "GLO12 persistence",
    "url": "https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024",
    "organisation": "Mercator Ocean",
    "organisation_url": "https://mercator-ocean.eu",
    "method": "Baseline",
    "forecast_type": "Deterministic",
    "initial_conditions": "GLO12",
    "resolution": "1/12°",
}

_CLIMATOLOGY = {
    "label": "Climatology",
    "url": "https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030",
    "organisation": "Mercator Ocean",
    "organisation_url": "https://mercator-ocean.eu",
    "method": "Baseline",
    "forecast_type": "GLORYS 1993-2019 day-of-year mean",
    "initial_conditions": "None",
    "resolution": "1/4°",
}

_GLORYS = {
    "label": "GLORYS reanalysis",
    "url": "https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030",
    "organisation": "Mercator Ocean",
    "organisation_url": "https://mercator-ocean.eu",
    "note": "A reanalysis that assimilates observations, shown as a reference, not a forecast.",
    "method": "Reference",
    "forecast_type": "Reanalysis",
    "initial_conditions": "Not applicable",
    "resolution": "1/4°",
}

_GLO12_ANALYSIS = {
    "label": "GLO12 analysis",
    "url": "https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024",
    "organisation": "Mercator Ocean",
    "organisation_url": "https://mercator-ocean.eu",
    "note": "An analysis that assimilates observations, shown as a reference, not a forecast.",
    "method": "Reference",
    "forecast_type": "Analysis",
    "initial_conditions": "Not applicable",
    "resolution": "1/12°",
}

_HCLIMREP = {
    "label": "HClimRep",
    "url": "https://arxiv.org/abs/2609.28601",
    "organisation": "AWI",
    "organisation_url": "https://www.awi.de/en/science/climate-sciences/climate-dynamics/team.html",
    "note": "Initialised with ERA5 reanalysis atmosphere, which is not available in real time.",
    "forecasts_run_by": "Authors",
    "method": "ML-based",
    "forecast_type": "Deterministic",
    "initial_conditions": "GLO12/ERA5",
    "resolution": "1/4°",
}

CHALLENGERS = {
    "glo12": _GLO12,
    "glo12_1_degree": {**_GLO12, "resolution": "1°"},
    "glonet": _GLONET,
    "glonet_1_degree": {**_GLONET, "resolution": "1°"},
    "wenhai": _WENHAI,
    "wenhai_1_degree": {**_WENHAI, "resolution": "1°"},
    "xihe": _XIHE,
    "xihe_1_degree": {**_XIHE, "resolution": "1°"},
    "langya": _LANGYA,
    "langya_1_degree": {**_LANGYA, "resolution": "1°"},
    "glo12_persistence": _GLO12_PERSISTENCE,
    "glo12_persistence_1_degree": {**_GLO12_PERSISTENCE, "resolution": "1°"},
    "hclimrep": _HCLIMREP,
    "hclimrep_1_degree": {**_HCLIMREP, "resolution": "1°"},
    "climatology": _CLIMATOLOGY,
    "climatology_1_degree": {**_CLIMATOLOGY, "resolution": "1°"},
    "glorys": _GLORYS,
    "glorys_1_degree": {**_GLORYS, "resolution": "1°"},
    "glorys_twelfth_degree": {**_GLORYS, "resolution": "1/12°"},
    "glo12_analysis": _GLO12_ANALYSIS,
}


def challenger_label(challenger_name: str) -> str:
    return CHALLENGERS.get(challenger_name, {}).get("label", challenger_name)


def challenger_category(challenger_name: str) -> str:
    method = CHALLENGERS.get(challenger_name, {}).get("method")
    return {"Baseline": "baseline", "Reference": "reference"}.get(method, "model")


def challenger_note(challenger_name: str) -> str | None:
    return CHALLENGERS.get(challenger_name, {}).get("note")


def challenger_forecasts_run_by(challenger_name: str) -> str:
    return CHALLENGERS.get(challenger_name, {}).get("forecasts_run_by", "Mercator Ocean")
