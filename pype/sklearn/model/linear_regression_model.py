from typing import Any

from sklearn.linear_model import LinearRegression

from pype.sklearn.model.sklearn_model import SklearnModel


class LinearRegressionModel(SklearnModel[LinearRegression]):
    def _init_model(self, args: dict[str, Any]) -> LinearRegression:
        return LinearRegression(**args)
