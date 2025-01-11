import tomli
from pandas import DataFrame, read_csv
from src.config.core import config


def get_version() -> str:
    """
    returns the version from pyproject.toml
    """
    with open("pyproject.toml", "rb") as f:
        pyproject_data = tomli.load(f)
    return pyproject_data["tool"]["poetry"]["version"]


def load_dataset(path: str, sep=",") -> DataFrame:
    """
    Load dataset from a file.
    """
    df = read_csv(path, sep=sep, usecols=config.model.vars)
    return df
