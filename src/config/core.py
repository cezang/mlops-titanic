from pathlib import Path
from typing import List, Any
import yaml
from pydantic import BaseModel, ValidationError

CONFIGPATH = Path().absolute().parent / "config.yml"

class ModelConfig(BaseModel):
    vars: List[str]
    vars_to_replace_category: List[str]
    categories_to_leave: List[str]
    replace_with: str
    string_to_na: str
    na_type: Any
    vars_to_float: List[str]
    var_to_extract_title: List[str]
    var_name_of_title: List[str]
    vars_na_to_mean: List[str]
    vars_na_to_mfrq: List[str]
    vars_to_map: List[str]
    dicts_to_map: List[dict]
    vars_to_freq_encode: List[str]

class Config(BaseModel):
    model: ModelConfig


class ConfigLoader:
    def __init__(self, config_path=None):
        # Default path to the configuration file
        self.config_path = Path(config_path or CONFIGPATH)
        self.config = self.load_config()

    def load_config(self):
        # Check if the file exists
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        # Load the YAML file
        with self.config_path.open("r") as file:
            raw_config = yaml.safe_load(file)
        
        # Validate data using Pydantic
        try:
            return Config(**raw_config)
        except ValidationError as e:
            raise ValueError(f"Invalid configuration: {e}")

    def get(self):
        return self.config