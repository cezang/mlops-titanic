from typing import List
from classification_model.version import __version__
from pandas import DataFrame, read_csv
from classification_model.config.core import (
    config,
    DATASET_DIR,
    TRAINED_MODEL_DIR,
)
from pathlib import Path
from sklearn.pipeline import Pipeline
import joblib


def load_dataset(file_name: str) -> DataFrame:
    """
    Load dataset from a file.
    """
    df = read_csv(
        Path(f"{DATASET_DIR}/{file_name}"),
        usecols=[config.model.target] + config.model.vars,
    )
    return df


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app.pipeline_name}_{__version__}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
