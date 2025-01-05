from sklearn.base import BaseEstimator, TransformerMixin
import re

class TitleExtractor(BaseEstimator, TransformerMixin):
    """
    A custom transformer for extracting titles from strings.
    """
    def __init__(self, variables=None, list_of_new_col_names=None):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        if not isinstance(list_of_new_col_names, list):
            raise ValueError("list_of_new_col_names should be a list")
        self.variables = variables
        self.list_of_new_col_names = list_of_new_col_names
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature, new_col_name in zip(self.variables, self.list_of_new_col_names):
            X[feature] = X[feature].apply(self._extract_title)
            X.rename(columns={feature: new_col_name}, inplace=True)
        return X

    def _extract_title(self, title):
        match = re.search(r',\s*([^,]+)\.', title)
        if match:
            return match.group(1).strip()
        else:
            return title.strip()
