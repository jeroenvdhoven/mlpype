from unittest.mock import MagicMock, call

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pytest import fixture

from mlpype.base.data.dataset import DataSet
from mlpype.spark.evaluate.spark_evaluator import SparkEvaluator
from tests.spark.utils import spark_session
from tests.utils import pytest_assert

spark_session


class TestSparkEvaluator:
    @fixture
    def pred_data(self) -> DataSet:
        return DataSet(df=MagicMock())

    @fixture
    def model(self, pred_data: DataSet) -> MagicMock:
        model = MagicMock()

        model.transform_for_evaluation.return_value = pred_data
        model.outputs = ["df"]
        return model

    def test_evalute(self, model: MagicMock, pred_data: DataSet):
        java_eval = MagicMock()
        metrics = ["a", "b"]
        evaluator = SparkEvaluator(java_eval, metrics)

        data = MagicMock()
        result = evaluator.evaluate(model, data)

        model.transform_for_evaluation.assert_called_once_with(data)

        assert java_eval.setMetricName.call_count == 2
        for i, metric in enumerate(metrics):
            assert java_eval.setMetricName.call_args_list[i] == call(metric)

        set_return = java_eval.setMetricName.return_value
        set_return.evaluate.assert_has_calls([call(pred_data["df"]), call(pred_data["df"])])

        expected = {
            "a": set_return.evaluate.return_value,
            "b": set_return.evaluate.return_value,
        }

        assert result == expected

    def test_evaluate_assert(self):
        java_eval = MagicMock()
        metrics = ["a", "b"]
        evaluator = SparkEvaluator(java_eval, metrics)

        model = MagicMock()
        multi_result_set = DataSet(x=MagicMock(), df=MagicMock())
        model.transform_for_evaluation.return_value = multi_result_set

        data = MagicMock()

        with pytest_assert(AssertionError, "Expect exactly 1 output from a SparkModel."):
            evaluator.evaluate(model, data)

    def test_evaluate_transform(self, model: MagicMock, pred_data: DataSet):
        java_eval = MagicMock()
        metrics = ["a", "b"]
        evaluator = SparkEvaluator(java_eval, metrics)

        pipe_transform = MagicMock()
        pipeline = MagicMock()
        pipeline.transform.return_value = pipe_transform

        data = MagicMock()
        result = evaluator.evaluate(model, data, pipeline=pipeline)

        pipeline.transform.assert_called_once_with(data)
        model.transform_for_evaluation.assert_called_once_with(pipe_transform)

        assert java_eval.setMetricName.call_count == 2
        for i, metric in enumerate(metrics):
            assert java_eval.setMetricName.call_args_list[i] == call(metric)

        set_return = java_eval.setMetricName.return_value
        set_return.evaluate.assert_has_calls([call(pred_data["df"]), call(pred_data["df"])])

        expected = {
            "a": set_return.evaluate.return_value,
            "b": set_return.evaluate.return_value,
        }

        assert result == expected

    def test_str(self, spark_session: SparkSession):  # needed for initialising RegressionEvaluator
        java_eval = RegressionEvaluator()
        metrics = ["a", "b"]
        evaluator = SparkEvaluator(java_eval, metrics)

        result = evaluator.__str__(indents=2)
        expected = "\t\tRegressionEvaluator: a, b"

        assert expected == result
