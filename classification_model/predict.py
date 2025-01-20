from classification_model.config.core import config
from classification_model.processing.data_menager import (
    load_pipeline
)
from classification_model.version import __version__
from classification_model.processing.validation import validate_input_data
from pandas import DataFrame

pipeline_filename = config.app.pipeline_name + "_" + __version__ + ".pkl"
trained_model = load_pipeline(file_name=pipeline_filename)


def make_prediction(*, input_data: DataFrame) -> dict:
    data = input_data.copy()
    data_validated, errors = validate_input_data(input_data=data)
    predictions = trained_model.predict(data_validated)
    results = {"predictions": None, "version": __version__, "errors": errors}

    if not errors:
        predictions = trained_model.predict(X=data_validated)
        results = {
            "predictions": predictions,
            "version": __version__,
            "errors": errors,
        }

    return results
