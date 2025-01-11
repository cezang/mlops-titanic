import pytest
from src.processing.data_menager import load_dataset
from src.config.core import config
from pandas import DataFrame


@pytest.fixture
def sample_dataset() -> DataFrame:
    return load_dataset(config.app.test_data_file)
