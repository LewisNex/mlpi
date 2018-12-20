from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names, pt_bounds):
        self.attribute_names = attribute_names
        self.pt_bounds = pt_bounds
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[X["pt"].between(*self.pt_bounds)][self.attribute_names].values

