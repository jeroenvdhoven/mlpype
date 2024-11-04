from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from pytest import fixture
from sklearn.linear_model import LinearRegression

from mlpype.base.data.dataset import DataSet
from mlpype.sklearn.model import LinearRegressionModel
from mlpype.sklearn.model.sklearn_model import SklearnModel


class Test_SklearnModel:
    @fixture
    def sklearn_model(self):
        return MagicMock()

    @fixture
    def model(self, sklearn_model: MagicMock):
        return LinearRegressionModel(inputs=["x"], outputs=["y"], model=sklearn_model, seed=43)

    def test_set_seed(self, model: SklearnModel):
        with patch("mlpype.sklearn.model.sklearn_model.np.random.seed") as mock_seed:
            model.set_seed()
        mock_seed.assert_called_once_with(43)

    def test_fit(self, model: SklearnModel, sklearn_model: MagicMock):
        ds1 = MagicMock()
        ds2 = MagicMock()
        model._fit(ds1, ds2)

        sklearn_model.fit.assert_called_once_with(ds1, ds2)

    def test_transform(self, model: SklearnModel, sklearn_model: MagicMock):
        ds1 = MagicMock()
        ds2 = MagicMock()
        model._transform(ds1, ds2)

        sklearn_model.predict.assert_called_once_with(ds1, ds2)

    def test_save(self, model: SklearnModel, sklearn_model: MagicMock):
        with patch("mlpype.sklearn.model.sklearn_model.JoblibSerialiser.serialise") as mock_serialise:
            folder = Path("folder")
            model._save(folder)

            mock_serialise.assert_called_once_with(sklearn_model, folder / model.SKLEARN_MODEL_FILE)

    def test_load(self, model: SklearnModel, sklearn_model: MagicMock):
        with patch(
            "mlpype.sklearn.model.sklearn_model.JoblibSerialiser.deserialise", return_value=sklearn_model
        ) as mock_deserialise:
            folder = Path("folder")
            inputs = ["x"]
            outputs = ["y"]
            result: SklearnModel = model._load(folder, inputs, outputs)

            mock_deserialise.assert_called_once_with(folder / model.SKLEARN_MODEL_FILE)
            assert result.model == sklearn_model
            assert result.inputs == inputs
            assert result.outputs == outputs

    def test_get_parameters_from_object(self):
        class DummyModel(SklearnModel[LinearRegression]):
            pass

        parser = MagicMock()
        model = DummyModel(inputs=["x"], outputs=["y"])

        with patch("mlpype.sklearn.model.sklearn_model.add_args_to_parser_for_class") as mock_add_args:
            model.get_parameters(parser)
            mock_add_args.assert_called_once_with(
                parser, LinearRegression, "model", [], excluded_args=["seed", "inputs", "outputs", "model"]
            )

    def test_get_parameters_from_class(self):
        class DummyModel(SklearnModel[LinearRegression]):
            pass

        parser = MagicMock()

        with patch("mlpype.sklearn.model.sklearn_model.add_args_to_parser_for_class") as mock_add_args:
            DummyModel.get_parameters(parser)
            mock_add_args.assert_called_once_with(
                parser, LinearRegression, "model", [], excluded_args=["seed", "inputs", "outputs", "model"]
            )

    def test_integration(self):
        model = LinearRegressionModel(["inputs"], ["outputs"])

        x = pd.DataFrame({"x": [1, 2, 3, 4], "y": [3, 5, 6, 7]})
        y = pd.DataFrame({"z": x["x"] * 1.5 - 2 * x["y"] + 2.4})
        data = DataSet(inputs=x.copy(), outputs=y.copy())

        model.fit(data)

        predictions = model.transform(data)

        assert isinstance(predictions["outputs"], np.ndarray)
        np.testing.assert_array_almost_equal(y, predictions["outputs"])

    def test_class_from_sklearn_model_class(self):
        klass = SklearnModel.class_from_sklearn_model_class(LinearRegression)

        assert klass.__name__ == "LinearRegressionModel"
        assert issubclass(klass, SklearnModel)

        annotated = klass._get_annotated_class()
        assert annotated == LinearRegression

    def test_from_sklearn_model_class(self):
        model = SklearnModel.from_sklearn_model_class(LinearRegression, ["inputs"], ["outputs"])

        assert model.__class__.__name__ == "LinearRegressionModel"
        assert isinstance(model, SklearnModel)

        annotated = model._get_annotated_class()
        assert annotated == LinearRegression

        x = pd.DataFrame({"x": [1, 2, 3, 4], "y": [3, 5, 6, 7]})
        y = pd.DataFrame({"z": x["x"] * 1.5 - 2 * x["y"] + 2.4})
        data = DataSet(inputs=x.copy(), outputs=y.copy())

        model.fit(data)

        predictions = model.transform(data)

        assert isinstance(predictions["outputs"], np.ndarray)
        np.testing.assert_array_almost_equal(y, predictions["outputs"])
