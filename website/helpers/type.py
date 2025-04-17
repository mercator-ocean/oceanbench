from pydantic import BaseModel


class ModelVariable(BaseModel):
    cf_name: str
    unit: str
    data: dict[str, float]


class ModelDepth(BaseModel):
    real_value: float
    variables: dict[str, ModelVariable]


class ModelScore(BaseModel):
    name: str
    depths: dict[str, ModelDepth]
