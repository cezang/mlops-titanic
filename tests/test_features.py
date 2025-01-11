from pandas import DataFrame
from src.processing.features import TitleExtractor
from src.config.core import config


def test_title_extractor(sample_dataset: DataFrame) -> None:
    transformer = TitleExtractor(
        variables=config.model.var_to_extract_title,
        list_of_new_col_names=config.model.var_name_of_title,
    )

    assert (
        sample_dataset.iloc[99]["name"]
        == "Klasen, Mrs. (Hulda Kristina Eugenia Lofqvist)"
    )

    assert transformer.fit_transform(sample_dataset).iloc[99]["title"] == "Mrs"
