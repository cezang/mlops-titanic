from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Any
from pandas import DataFrame
from classification_model.processing.features import CastNaOnString
from classification_model.config.core import config


class InputDataSchema(BaseModel):
    pclass: Optional[int]
    name: Optional[str]
    sex: Optional[str]
    age: Optional[str]
    sibsp: Optional[int]
    parch: Optional[int]
    ticket: Optional[str]
    fare: Optional[float]
    cabin: Optional[str]
    embarked: Optional[str]
    boat: Optional[str]
    body: Optional[str]
    home_dest: Optional[str] = Field(None, alias="home.dest")  # alias "home.dest"

    class Config:
        # Setting to allow Pydantic to handle aliases during JSON conversion
        allow_population_by_field_name = True
        # Additional options that may help in alias mapping
        anystr_strip_whitespace = True


class MultipleDataInputs(BaseModel):
    inputs: List[InputDataSchema]

class PredictionResult(BaseModel):
    predictions: list
    version: str
    errors: dict = {}

def validate_input_data(*, input_data: DataFrame) -> tuple[Any, str | None]:
    """Validate input data, return input data and errors"""
    errors = None

    transformer = CastNaOnString(
        variables=config.model.vars_to_validate,
        string=config.model.string_to_na,
        na=config.model.na_type,
    )
    data = transformer.fit_transform(input_data.copy())
    # return data with default columns used in training
    return_data = input_data[config.model.vars]
    # we validate all the data
    try:
        MultipleDataInputs(inputs=data.to_dict(orient="records"))
    except ValidationError as error:
        errors = error.json()
    return return_data, errors
