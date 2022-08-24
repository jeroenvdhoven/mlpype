from typing import Any

from sklearn.linear_model import LogisticRegression

from pype.sklearn.model.sklearn_model import SklearnModel


class LogisticRegressionModel(SklearnModel[LogisticRegression]):
    def _init_model(self, args: dict[str, Any]) -> LogisticRegression:
        return LogisticRegression(**args)
