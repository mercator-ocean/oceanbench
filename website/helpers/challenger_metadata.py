# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

CHALLENGERS = {
    "glo12": {
        "label": "GLO12",
        "url": "https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024",
        "organisation": "Mercator Ocean",
        "organisation_url": "https://mercator-ocean.eu",
        "method": "Physics-based",
        "forecast_type": "Deterministic",
        "initial_conditions": "IFS HRES",
        "resolution": "1/12\u00b0",
    },
    "glonet": {
        "label": "GLONET",
        "url": "https://glonet.lab.dive.edito.eu",
        "organisation": "Mercator Ocean",
        "organisation_url": "https://mercator-ocean.eu",
        "method": "ML-based",
        "forecast_type": "Deterministic",
        "initial_conditions": "GLO12",
        "resolution": "1/4\u00b0",
    },
    "wenhai": {
        "label": "WenHai",
        "url": "https://www.nature.com/articles/s41467-025-57389-2",
        "organisation": "DOMES",
        "organisation_url": "http://iaos.ouc.edu.cn/DeepOceanMultispheresandEarthSystem/list.htm",
        "method": "ML-based",
        "forecast_type": "Deterministic",
        "initial_conditions": "GLO12/IFS",
        "resolution": "1/12\u00b0",
    },
    "xihe": {
        "label": "XiHe",
        "url": "https://arxiv.org/abs/2402.02995",
        "organisation": "NUDT",
        "organisation_url": "https://english.nudt.edu.cn/",
        "method": "ML-based",
        "forecast_type": "Deterministic",
        "initial_conditions": "GLO12/IFS",
        "resolution": "1/12\u00b0",
    },
}

KNOWN_CHALLENGERS = list(CHALLENGERS.keys())
