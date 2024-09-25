"""Provides classes for sklearn models.

For sklearn models not already configured here, you can use the SklearnModel class to quickly incorporate your model.
"""
from sklearn.linear_model import LinearRegression, LogisticRegression

from .sklearn_base_type import SklearnModelBaseType
from .sklearn_model import SklearnModel

LinearRegressionModel = SklearnModel.class_from_sklearn_model_class(LinearRegression)
LogisticRegressionModel = SklearnModel.class_from_sklearn_model_class(LogisticRegression)

__all__ = [
    "SklearnModel",
    "SklearnModelBaseType",
    "LinearRegressionModel",
    "LogisticRegressionModel",
]
