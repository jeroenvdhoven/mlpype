from typing import Any, Dict

from sklearn.linear_model import LogisticRegression

from mlpype.sklearn.model.sklearn_model import SklearnModel


class LogisticRegressionModel(SklearnModel[LogisticRegression]):
    def _init_model(self, args: Dict[str, Any]) -> LogisticRegression:
        return LogisticRegression(**args)
