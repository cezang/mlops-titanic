from sklearn.base import BaseEstimator, TransformerMixin
import re

class TitleExtractor(BaseEstimator, TransformerMixin):
    """
    A custom transformer for extracting titles from strings.
    """
    def __init__(self):
        """
        Initialize the transformer.
        """
        pass
    
    def fit(self, X, y=None):
        """
        Fit the transformer. This transformer does not need fitting, so we just return self.
        
        Args:
            X (array-like): The data to fit.
            y (array-like, optional): The target values (ignored).

        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Transform the input data by extracting titles.

        Args:
            X (pandas col): The column to transform.

        Returns:
            pd.Series: The transformed data with extracted titles.
        """
        return X.apply(self._extract_title)

    def _extract_title(self, title):
        """
        Extracts the title from the given string.

        Args:
            title (str): The string containing the title.

        Returns:
            str: The extracted title.
        """
        match = re.search(r',\s*([^,]+)\.', title)
        if match:
            return match.group(1).strip()
        else:
            return title.strip()
