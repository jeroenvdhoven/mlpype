"""Provides a class for sklearn linear regression models."""
from typing import Any, Dict

from sklearn.linear_model import LinearRegression

from mlpype.sklearn.model.sklearn_model import SklearnModel


class LinearRegressionModel(SklearnModel[LinearRegression]):
    """A class for sklearn linear regression models."""

    def _init_model(self, args: Dict[str, Any]) -> LinearRegression:
        return LinearRegression(**args)
