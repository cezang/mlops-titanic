import pytest
from classification_model.processing.data_menager import load_dataset
from classification_model.processing.features import CastNaOnString
from classification_model.config.core import config
from pandas import DataFrame


@pytest.fixture
def sample_dataset() -> DataFrame:
    return load_dataset(config.app.test_data_file_with_target)


@pytest.fixture
def sample_dataset_after_na() -> DataFrame:
    transformerna = CastNaOnString(
        variables=config.model.vars,
        string=config.model.string_to_na,
        na=config.model.na_type,
    )
    return transformerna.fit_transform(
        load_dataset(config.app.test_data_file_with_target)
    )
