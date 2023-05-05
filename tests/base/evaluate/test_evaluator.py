from unittest.mock import MagicMock

from pytest import fixture

from mlpype.base.data import DataSet
from mlpype.base.evaluate.evaluator import Evaluator


class Test_Evaluator:
    @fixture
    def data(self):
        return DataSet[int](a=1, b=2, c=3, d=9)

    @fixture
    def predictions(self):
        return DataSet[int](c=4, d=5)

    @fixture
    def model(self, predictions):
        model = MagicMock()
        model.transform.return_value = predictions
        model.outputs = list(predictions.keys())
        return model

    @fixture
    def evaluator(self):
        return Evaluator(
            {
                "function_a": MagicMock(),
                "function_b": MagicMock(),
            }
        )

    def test(self, data: DataSet[int], model: MagicMock, evaluator: Evaluator):
        result = evaluator.evaluate(model, data)

        expected = {name: value.return_value for name, value in evaluator.functions.items()}
        assert expected == result
        model.transform.assert_called_once_with(data)

        for func in evaluator.functions.values():
            func.assert_called_once_with(3, 9, 4, 5)

    def test_with_pipeline(self, data: DataSet[int], model: MagicMock, evaluator: Evaluator):
        transform_data = DataSet[int](c=41, d=51)

        pipeline = MagicMock()
        pipeline.transform.return_value = transform_data
        result = evaluator.evaluate(model, data, pipeline)

        expected = {name: value.return_value for name, value in evaluator.functions.items()}
        assert expected == result
        pipeline.transform.assert_called_once_with(data)
        model.transform.assert_called_once_with(transform_data)

        for func in evaluator.functions.values():
            func.assert_called_once_with(41, 51, 4, 5)
