from pathlib import Path
import importlib.resources
from typing import List, Any, Optional
import yaml
from pydantic import BaseModel, ValidationError

CONFIGPATH = Path(__file__).parent.parent / "config.yml"
DATASET_DIR = Path(__file__).parent.parent / "data"
TRAINED_MODEL_DIR = (
    Path(__file__).parent.parent / "trained_models"
)
PACKAGE = "classification_model"
RESOURCE = "config.yml"

class AppConfig(BaseModel):
    train_data_file: str
    test_data_file: str
    test_data_file_with_target: str
    random_state: int
    pipeline_name: str


class ModelConfig(BaseModel):
    vars: List[str]
    vars_to_replace_category: List[str]
    categories_to_leave: List[str]
    replace_with: str
    string_to_na: str
    na_type: Optional[Any]
    vars_to_float: List[str]
    var_to_extract_title: List[str]
    var_name_of_title: List[str]
    vars_na_to_mean: List[str]
    vars_na_to_mfreq: List[str]  # Corrected field name
    vars_to_map: List[str] = []  # Default empty list
    dicts_to_map: List[dict] = []  # Default empty list
    vars_to_freq_encode: List[str] = []  # Default empty list
    n_estimators: int
    test_size: float
    target: str
    vars_to_validate: List[str]


class Config(BaseModel):
    app: AppConfig
    model: ModelConfig


def load_config(config_path: Path = CONFIGPATH) -> Config:
    # Check if the file exists
    # if not config_path.exists():
    #     raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load the YAML file
    with importlib.resources.open_text(PACKAGE, RESOURCE) as file:
        raw_config: dict = yaml.safe_load(file)
    # with config_path.open("r") as file:
    #     raw_config: dict = yaml.safe_load(file)

    # Validate data using Pydantic
    try:
        return Config(**raw_config)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}")


# Usage
config = load_config()
