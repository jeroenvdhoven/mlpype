"""Provides a class for sklearn linear regression models."""

from sklearn.linear_model import LinearRegression

from mlpype.sklearn.model.sklearn_model import SklearnModel


class LinearRegressionModel(SklearnModel[LinearRegression]):
    """A class for sklearn linear regression models."""
