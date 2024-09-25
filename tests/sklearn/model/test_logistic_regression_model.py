from unittest.mock import MagicMock

from sklearn.linear_model import LogisticRegression

from mlpype.sklearn.model import LogisticRegressionModel


class Test_LogisticRegressionModel:
    def test_init_new_model(self):
        model = LogisticRegressionModel(["x"], ["y"])

        assert isinstance(model.model, LogisticRegression)

    def test_init_existing_model(self):
        sklearn_model = MagicMock()
        model = LogisticRegressionModel(["x"], ["y"], model=sklearn_model)

        assert model.model == sklearn_model
