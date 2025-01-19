from src.config.core import config
from src.processing.data_menager import load_pipeline, get_version
from src.processing.validation import validate_input_data
from pandas import DataFrame

pipeline_filename = config.app.pipeline_name + "_" + get_version() + ".pkl"
trained_model = load_pipeline(file_name=pipeline_filename)


def make_prediction(*, input_data: DataFrame) -> dict:
    data = input_data.copy()
    data_validated, errors = validate_input_data(input_data=data)
    predictions = trained_model.predict(data_validated)
    results = {"predictions": None, "version": get_version(), "errors": errors}

    if not errors:
        predictions = trained_model.predict(X=data_validated)
        results = {
            "predictions": predictions,
            "version": get_version(),
            "errors": errors,
        }

    return results
