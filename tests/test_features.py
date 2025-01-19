from pandas import DataFrame
from classification_model.processing.features import (
    TitleExtractor,
    ReplaceCatogories,
    CastNaOnString,
    Mapper,
    CastType,
)
from classification_model.config.core import config


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


def test_replace_categories(sample_dataset: DataFrame) -> None:
    transformer = ReplaceCatogories(
        variables=config.model.vars_to_replace_category,
        list_of_category_to_leave=config.model.categories_to_leave,
        replace_with=config.model.replace_with,
    )

    assert sample_dataset.iloc[1]["boat"] == "C"
    assert transformer.fit_transform(sample_dataset).iloc[99]["boat"] == "?"
    assert transformer.fit_transform(sample_dataset).iloc[1]["boat"] == "any"


def test_cast_na_on_string(sample_dataset: DataFrame) -> None:
    transformer = CastNaOnString(
        variables=config.model.vars,
        string=config.model.string_to_na,
        na=config.model.na_type,
    )

    assert sample_dataset.iloc[99]["boat"] == "?"
    assert transformer.fit_transform(sample_dataset).iloc[99]["boat"] is None


def test_mapper(sample_dataset_after_na: DataFrame) -> None:
    transformer = Mapper(
        variables=config.model.vars_to_map, mappings=config.model.dicts_to_map
    )
    assert sample_dataset_after_na.iloc[99]["boat"] is None
    assert sample_dataset_after_na.iloc[99]["sex"] == "female"

    assert (
        transformer.fit_transform(sample_dataset_after_na).iloc[99]["boat"]
        == 0
    )
    assert (
        transformer.fit_transform(sample_dataset_after_na).iloc[99]["sex"] == 0
    )


def test_cast_type(sample_dataset_after_na: DataFrame) -> None:
    transformer = CastType(variables=config.model.vars_to_float, dtype=float)
    assert isinstance(sample_dataset_after_na.iloc[99]["age"], str)
    assert isinstance(
        transformer.fit_transform(sample_dataset_after_na).iloc[99]["age"],
        float,
    )
