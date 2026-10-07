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
    "initial_conditions": "GLO12 nowcast",
    "resolution": "1/12°",
}

_HCLIMREP = {
    "label": "HClimRep",
    "url": "https://hclimrep-project.de/",
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
}


def challenger_label(challenger_name: str) -> str:
    return CHALLENGERS.get(challenger_name, {}).get("label", challenger_name)


def challenger_category(challenger_name: str) -> str:
    method = CHALLENGERS.get(challenger_name, {}).get("method")
    return "baseline" if method == "Baseline" else "model"


def challenger_note(challenger_name: str) -> str | None:
    return CHALLENGERS.get(challenger_name, {}).get("note")


def challenger_forecasts_run_by(challenger_name: str) -> str:
    return CHALLENGERS.get(challenger_name, {}).get("forecasts_run_by", "Mercator Ocean")
