from sklearn.base import BaseEstimator, TransformerMixin
import re
from pandas import DataFrame
from typing import Any


class TitleExtractor(BaseEstimator, TransformerMixin):
    """
    A custom transformer for extracting titles from strings.
    """

    def __init__(
        self, variables: list[str], list_of_new_col_names: list[str]
    ) -> None:
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        if not isinstance(list_of_new_col_names, list):
            raise ValueError("list_of_new_col_names should be a list")
        self.variables = variables
        self.list_of_new_col_names = list_of_new_col_names

    def fit(self, X: DataFrame, y: None = None) -> "TitleExtractor":
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        X = X.copy()
        for feature, new_col_name in zip(
            self.variables, self.list_of_new_col_names
        ):
            X[feature] = X[feature].apply(self._extract_title)
            X.rename(columns={feature: new_col_name}, inplace=True)
        return X

    def _extract_title(self, title: str) -> str:
        match = re.search(r",\s*([^,]+)\.", title)
        if match:
            return match.group(1).strip()
        else:
            return title.strip()


class ReplaceCatogories(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        variables: list[str],
        list_of_category_to_leave: list[list[str]],
        replace_with: str = "any",
    ) -> None:
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        if not isinstance(list_of_category_to_leave, list):
            raise ValueError("list_of_category_to_leave should be a list")
        self.variables = variables
        self.list_of_category_to_leave = list_of_category_to_leave
        self.replace_with = replace_with

    def fit(self, X: DataFrame, y: None = None) -> "ReplaceCatogories":
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        X = X.copy()
        for feature, category in zip(
            self.variables, self.list_of_category_to_leave
        ):
            X[feature] = X[feature].apply(
                lambda x: x if x in category else self.replace_with
            )
        return X


class CastNaOnString(BaseEstimator, TransformerMixin):
    def __init__(self, variables: list[str], string: str, na: Any) -> None:
        self.variables = variables
        self.string = string
        self.na = na

    def fit(self, X: DataFrame, y: None = None) -> "CastNaOnString":
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].replace(self.string, self.na)
        return X


class Mapper(BaseEstimator, TransformerMixin):
    """
    Maps values of specified variables using provided mappings.
    """

    def __init__(self, variables: list[str], mappings: list[dict]) -> None:
        """
        Initializes with variables and mappings.
        """
        self.variables = variables
        self.mappings = mappings

    def fit(self, X: DataFrame, y: None = None) -> "Mapper":
        """
        No fitting necessary.
        """
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """
        Maps values of specified variables.
        """
        X = X.copy()
        for feature, mapping in zip(self.variables, self.mappings):
            X[feature] = X[feature].map(mapping)
        return X


class CastType(BaseEstimator, TransformerMixin):
    """
    Casts specified variables to specified type.
    """

    def __init__(self, variables: list[str], dtype: type) -> None:
        """
        Initializes with variables and dtype.
        """
        self.variables = variables
        self.dtype = dtype

    def fit(self, X: DataFrame, y: None = None) -> "CastType":
        """
        No fitting necessary.
        """
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """
        Casts specified variables to specified type.
        """
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].astype(self.dtype)
        return X
