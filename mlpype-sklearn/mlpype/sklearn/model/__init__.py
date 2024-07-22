"""Provides classes for sklearn models.

For sklearn models not already configured here, you can use the SklearnModel class to quickly incorporate your model.
"""
from .linear_regression_model import LinearRegressionModel
from .logistic_regression_model import LogisticRegressionModel
from .sklearn_base_type import SklearnModelBaseType
from .sklearn_model import SklearnModel
