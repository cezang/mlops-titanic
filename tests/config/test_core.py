import pytest
from unittest.mock import patch, mock_open
from pathlib import Path
import yaml
from src.config.core import ConfigLoader
from numpy import NaN


@pytest.fixture
def valid_config_dict() -> dict:
    """Fixture providing valid configuration data."""
    return {
        "model": {
            "vars": ["var1", "var2"],
            "vars_to_replace_category": ["var3"],
            "categories_to_leave": ["category1", "category2"],
            "replace_with": "unknown",
            "string_to_na": "N/A",
            "na_type": NaN,
            "vars_to_float": ["var4"],
            "var_to_extract_title": ["var5"],
            "var_name_of_title": ["title"],
            "vars_na_to_mean": ["var6"],
            "vars_na_to_mfrq": ["var7"],
            "vars_to_map": ["var8"],
            "dicts_to_map": [{"key1": "value1"}],
            "vars_to_freq_encode": ["var9"],
        }
    }


def test_load_config_valid_file(valid_config_dict: dict) -> None:
    """Test loading a valid YAML file."""
    mock_yaml = yaml.dump(valid_config_dict)  # Convert the dictionary to YAML

    # Mocking Path.exists and Path.open
    with (
        patch.object(Path, "exists", return_value=True),
        patch("pathlib.Path.open", mock_open(read_data=mock_yaml)),
    ):
        loader = ConfigLoader(config_path="fake/path/to/config.yaml")
        config = loader.get()

    assert config.model.vars == ["var1", "var2"]
    assert config.model.replace_with == "unknown"


def test_load_config_missing_file() -> None:
    """Test handling a missing configuration file."""
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(
            FileNotFoundError, match="Configuration file not found"
        ):
            ConfigLoader(config_path="fake/path/to/config.yaml")


def test_load_config_validation_error() -> None:
    """Test handling validation errors from Pydantic."""
    invalid_data = {"model": {"vars": "not_a_list"}}  # Invalid data format
    invalid_yaml = yaml.dump(invalid_data)

    # Mocking Path.exists and Path.open
    with (
        patch.object(Path, "exists", return_value=True),
        patch("pathlib.Path.open", mock_open(read_data=invalid_yaml)),
    ):
        with pytest.raises(ValueError, match="Invalid configuration"):
            ConfigLoader(config_path="fake/path/to/config.yaml")
