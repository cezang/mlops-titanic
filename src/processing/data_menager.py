import tomli
from typing import Any
from pandas import DataFrame, read_csv
from src.config.core import config, DATASET_DIR
from pathlib import Path


def get_version() -> Any:
    """
    returns the version from pyproject.toml
    """
    with open("pyproject.toml", "rb") as f:
        pyproject_data = tomli.load(f)
    return pyproject_data["tool"]["poetry"]["version"]


def load_dataset(file_name: str) -> DataFrame:
    """
    Load dataset from a file.
    """
    df = read_csv(
        Path(f"{DATASET_DIR}/{file_name}"), usecols=config.model.vars
    )
    return df


print(load_dataset(config.app.test_data_file).iloc[99]["name"])
