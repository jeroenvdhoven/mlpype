from typing import Any, Dict

from sklearn.linear_model import LinearRegression

from mlpype.sklearn.model.sklearn_model import SklearnModel


class LinearRegressionModel(SklearnModel[LinearRegression]):
    def _init_model(self, args: Dict[str, Any]) -> LinearRegression:
        return LinearRegression(**args)
