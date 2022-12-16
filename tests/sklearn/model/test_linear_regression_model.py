from unittest.mock import MagicMock

from sklearn.linear_model import LinearRegression

from pype.sklearn.model.linear_regression_model import LinearRegressionModel


class Test_LinearRegressionModel:
    def test_init_new_model(self):
        model = LinearRegressionModel(["x"], ["y"])

        assert isinstance(model.model, LinearRegression)

    def test_init_existing_model(self):
        sklearn_model = MagicMock()
        model = LinearRegressionModel(["x"], ["y"], model=sklearn_model)

        assert model.model == sklearn_model
