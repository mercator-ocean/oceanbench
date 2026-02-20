# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pydantic import BaseModel


class ModelVariable(BaseModel):
    cf_name: str
    unit: str
    data: dict[str, float]


class ModelDepth(BaseModel):
    variables: dict[str, ModelVariable]


class ModelScore(BaseModel):
    name: str
    depths: dict[str, ModelDepth]
