from unittest.mock import MagicMock, call

import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pytest import fixture

from mlpype.base.data.dataset import DataSet
from mlpype.spark.evaluate.spark_evaluator import SparkEvaluator
from mlpype.spark.model.linear_spark_model import LinearSparkModel
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

        model.transform.return_value = pred_data
        model.outputs = ["df"]
        return model

    def test_evalute(self, model: MagicMock, pred_data: DataSet):
        java_eval = MagicMock()
        metrics = ["a", "b"]
        evaluator = SparkEvaluator(java_eval, metrics)

        data = MagicMock()
        result = evaluator.evaluate(model, data)

        model.transform.assert_called_once_with(data)

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
        model.transform.return_value = multi_result_set

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
        model.transform.assert_called_once_with(pipe_transform)

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

    def test_evaluate_cache(self, model: MagicMock, pred_data: DataSet):
        java_eval = MagicMock()
        metrics = ["a", "b"]
        evaluator = SparkEvaluator(java_eval, metrics, set_cache=True)

        pipe_transform = MagicMock()
        pipeline = MagicMock()
        pipeline.transform.return_value = pipe_transform

        data = MagicMock()
        result = evaluator.evaluate(model, data, pipeline=pipeline)

        pipeline.transform.assert_called_once_with(data)
        model.transform.assert_called_once_with(pipe_transform)

        assert java_eval.setMetricName.call_count == 2
        for i, metric in enumerate(metrics):
            assert java_eval.setMetricName.call_args_list[i] == call(metric)

        pred_data["df"].cache.assert_called_once()
        preds = pred_data["df"].cache.return_value

        set_return = java_eval.setMetricName.return_value
        set_return.evaluate.assert_has_calls([call(preds), call(preds)])

        expected = {
            "a": set_return.evaluate.return_value,
            "b": set_return.evaluate.return_value,
        }

        assert result == expected

    def test_integration(self, spark_session: SparkSession):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [4, 5, 6],
            }
        )
        va = VectorAssembler(inputCols=["x"], outputCol="features")

        spark_df = spark_session.createDataFrame(df)
        spark_df = va.transform(spark_df)
        dataset = DataSet(df=spark_df)

        model = LinearSparkModel(
            inputs=["df"],
            outputs=["df"],
            featuresCol="features",
            labelCol="y",
        )
        model.fit(dataset)
        java_eval = RegressionEvaluator(labelCol="y")

        evaluator = SparkEvaluator(java_eval, ["mae", "mse"])
        result = evaluator.evaluate(model, dataset)

        assert "mae" in result
        assert "mse" in result
        for value in result.values():
            assert isinstance(value, float)
            assert value < 0.0001  # near perfect prediction is possible here.

    def test_str(self, spark_session: SparkSession):  # needed for initialising RegressionEvaluator
        java_eval = RegressionEvaluator()
        metrics = ["a", "b"]
        evaluator = SparkEvaluator(java_eval, metrics)

        result = evaluator.__str__(indents=2)
        expected = "\t\tRegressionEvaluator: a, b"

        assert expected == result
