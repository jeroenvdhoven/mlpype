"""Provides a class for sklearn logistic regression models."""

from sklearn.linear_model import LogisticRegression

from mlpype.sklearn.model.sklearn_model import SklearnModel


class LogisticRegressionModel(SklearnModel[LogisticRegression]):
    """A class for sklearn logistic regression models."""
