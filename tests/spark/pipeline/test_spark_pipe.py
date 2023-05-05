import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
from pandas.testing import assert_frame_equal
from pyspark.ml import Estimator, Transformer
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql import SparkSession
from pytest import fixture

from mlpype.base.data.dataset import DataSet
from mlpype.spark.pipeline.spark_pipe import SparkPipe
from tests.spark.utils import spark_session
from tests.utils import pytest_assert

spark_session


class DummyEstimator(Estimator):
    def _fit(self, data: Any):
        pass

    def _transform(self):
        pass


class DummyTransformer(Transformer):
    def _transform(self):
        pass


class Test_SparkPipe:
    @fixture
    def data(self) -> DataSet:
        return DataSet(x=MagicMock(), z=MagicMock())

    def test_fit_assert(self, data: DataSet):
        class Dummy:
            pass

        pipe = SparkPipe("test", Dummy, ["x"], ["y"])

        with pytest_assert(
            ValueError, f"In a SparkPipe, the operator must be a Transformer or Estimator. Got: {Dummy}"
        ):
            pipe.fit(data)

    def test_fit_transformer(self, data: DataSet):

        pipe = SparkPipe("test", DummyTransformer, ["x"], ["y"])

        pipe.fit(data)
        assert pipe.operator == pipe.fitted

    def test_fit_estimator(self, data: DataSet):

        pipe = SparkPipe("test", DummyEstimator, ["x"], ["y"])

        with patch.object(DummyEstimator, "fit") as mock_fit:
            pipe.fit(data)
            mock_fit.assert_called_once_with(data["x"])

    def test_transform(self, data: DataSet):
        pipe = SparkPipe("test", DummyEstimator, ["x"], ["y"])

        pipe.fitted = MagicMock()

        assert "y" not in data
        result = pipe.transform(data)
        assert "y" in result

        pipe.fitted.transform.assert_called_once_with(data["x"])
        assert result["y"] == pipe.fitted.transform.return_value

    def test_transform_in_inference(self, data: DataSet):
        pipe = SparkPipe("test", DummyEstimator, ["x"], ["y"], skip_on_inference=True)

        pipe.fitted = MagicMock()

        assert "y" not in data
        result = pipe.transform(data, is_inference=True)
        assert "y" not in result

        pipe.fitted.transform.assert_not_called()

    def test_set_state(self):
        warnings.warn("SparkPipe's __set_state__ is not tested atm.")

    def test_get_state(self):
        pipe = SparkPipe("test", DummyEstimator, ["x"], ["y"])

        state = pipe.__getstate__()

        assert "fitted" not in state
        assert "operator" not in state

    def test_get_state_from_fitted(self, data: DataSet):
        pipe = SparkPipe("test", DummyEstimator, ["x"], ["y"])
        pipe.fit(data)

        state = pipe.__getstate__()

        assert "fitted" not in state
        assert "operator" not in state

    def test_str(self):
        # just test if it runs.
        pipe = SparkPipe("test", DummyEstimator, ["x"], ["y"])
        str(pipe)

    def test_integration(self, spark_session: SparkSession):
        x = pd.DataFrame({"col1": [1, 2, 3, 4]})
        spark_x = VectorAssembler(inputCols=["col1"], outputCol="transformed").transform(
            spark_session.createDataFrame(x)
        )
        data = DataSet(x=spark_x, z=MagicMock())

        pipe = SparkPipe("test", MinMaxScaler, ["x"], ["y"], kw_args={"inputCol": "transformed", "outputCol": "scaled"})
        pipe.fit(data)

        assert "y" not in data
        transformed = pipe.transform(data)
        assert "y" in transformed
        transformed_df = transformed["y"].toPandas()

        expected = pd.DataFrame(
            {"col1": [1, 2, 3, 4], "transformed": [(1,), (2,), (3,), (4,)], "scaled": [(0,), (1 / 3,), (2 / 3,), (1,)]}
        )

        assert_frame_equal(expected, transformed_df)
