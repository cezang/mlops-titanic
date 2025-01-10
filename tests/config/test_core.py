import pytest
from unittest.mock import patch, mock_open
from pathlib import Path
from pydantic import ValidationError
from src.config.core import ConfigLoader, Config, CONFIGPATH


class TestConfigLoader:
    """Tests for the ConfigLoader class"""

    @pytest.fixture
    def valid_config_data(self):
        """Fixture providing valid YAML configuration data"""
        return """
        key1: value1
        key2: value2
        """

    @pytest.fixture
    def invalid_config_data(self):
        """Fixture providing invalid YAML configuration data"""
        return """
        key1: value1
        key2: 123456  # Assuming Config doesn't accept an integer for this key
        """

    def test_load_config_success(self, valid_config_data):
        """Test successful configuration loading"""
        with patch("pathlib.Path.open", mock_open(read_data=valid_config_data)):
            with patch("pathlib.Path.exists", return_value=True):
                loader = ConfigLoader("test_config.yaml")
                config = loader.get()
                assert config.key1 == "value1"
                assert config.key2 == "value2"

    def test_load_config_file_not_found(self):
        """Test when the configuration file does not exist"""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Configuration file not found:"):
                ConfigLoader("nonexistent_config.yaml")

    def test_load_config_invalid_data(self, invalid_config_data):
        """Test invalid data validation"""
        with patch("pathlib.Path.open", mock_open(read_data=invalid_config_data)):
            with patch("pathlib.Path.exists", return_value=True):
                with pytest.raises(ValueError, match="Invalid configuration:"):
                    ConfigLoader("invalid_config.yaml")

    def test_default_config_path(self, valid_config_data):
        """Test using the default CONFIGPATH"""
        with patch("pathlib.Path.open", mock_open(read_data=valid_config_data)):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("src.config.my_config.CONFIGPATH", "default_config.yaml"):
                    loader = ConfigLoader()
                    config = loader.get()
                    assert config.key1 == "value1"
                    assert config.key2 == "value2"
