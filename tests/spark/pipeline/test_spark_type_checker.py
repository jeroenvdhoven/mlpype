from typing import Dict, List, Union
from unittest.mock import MagicMock, call, patch

import pandas as pd
from pandas.testing import assert_frame_equal
from pydantic import create_model
from pyspark.sql import SparkSession
from pytest import mark

from pype.spark.pipeline.spark_type_checker import SparkData, SparkTypeChecker
from tests.spark.utils import spark_session
from tests.utils import pytest_assert

spark_session


class Test_SparkData:
    def test_convert(self, spark_session: SparkSession):
        data_type = {
            "a": (Union[List[int], Dict[str or int, int]], ...),
            "b": (Union[List[str], Dict[str or int, str]], ...),
        }

        CustomSparkData = create_model("CustomSparkData", **data_type, __base__=SparkData)

        data_obj = CustomSparkData(a=[1, 2, 3], b=["a", "2", "f"])

        converted = data_obj.convert()

        assert converted.columns == ["a", "b"]
        converted_df: pd.DataFrame = converted.toPandas()

        expected = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["a", "2", "f"],
            }
        )

        assert_frame_equal(expected, converted_df)

    def test_to_model(self, spark_session: SparkSession):
        data_type = {
            "a": (Union[List[int], Dict[str or int, int]], ...),
            "b": (Union[List[str], Dict[str or int, str]], ...),
        }
        CustomSparkData = create_model("CustomSparkData", **data_type, __base__=SparkData)

        df = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "2", "f"]})
        spark_df = spark_session.createDataFrame(df)

        result = CustomSparkData.to_model(spark_df)

        assert result.a == [1, 2, 3]
        assert result.b == ["a", "2", "f"]


class Test_SparkTypeChecker:
    def test_fit(self):
        data = MagicMock()
        dtypes = {"a": int, "b": float}
        data.dtypes = dtypes
        checker = SparkTypeChecker()

        with patch.object(checker, "_convert_dtypes") as mock_convert:
            result = checker.fit(data)

        mock_convert.assert_called_once_with(dtypes)
        assert result == checker

    def test_convert_dtypes(self):
        checker = SparkTypeChecker()
        data = {"a": "2", "b": "9"}
        returns = [2, 3]

        with patch.object(checker, "_convert_dtype", side_effect=returns) as mock_convert:
            result = checker._convert_dtypes(data)

        mock_convert.assert_has_calls([call("2"), call("9")])

        assert result == {"a": 2, "b": 3}

    @mark.parametrize(
        ["string_type", "expected"],
        [
            ["str", str],
            ["string", str],
            ["int", int],
            ["bigint", int],
            ["float", float],
            ["double", float],
            ["bool", bool],
        ],
    )
    def test_convert_dtype(self, string_type: str, expected: type):
        checker = SparkTypeChecker()
        result = checker._convert_dtype(string_type)

        assert result == (string_type, expected)

    def test_convert_dtype_assertion(self):
        checker = SparkTypeChecker()
        with pytest_assert(ValueError, "- not supported"):
            checker._convert_dtype("-")

    def test_transform_assert_missing_columns(self, spark_session: SparkSession):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "2", "f"]})
        spark_df = spark_session.createDataFrame(df)

        missing_df = pd.DataFrame({"a": [1, 2, 3]})
        spark_missing_df = spark_session.createDataFrame(missing_df)

        checker = SparkTypeChecker()
        checker.fit(spark_df)

        with pytest_assert(AssertionError, "`b` is missing from the dataset"):
            checker.transform(spark_missing_df)

    def test_transform_check_dtype(self, spark_session: SparkSession):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "2", "f"]})
        spark_df = spark_session.createDataFrame(df)

        missing_df = pd.DataFrame({"a": [1, 2, 3], "b": [2.2, 3.4, 5.6]})
        spark_missing_df = spark_session.createDataFrame(missing_df)

        checker = SparkTypeChecker()
        checker.fit(spark_df)

        with pytest_assert(AssertionError, "Dtypes did not match up for col b: Expected string, got double"):
            checker.transform(spark_missing_df)

    def test_transform_pass(self, spark_session: SparkSession):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "2", "f"]})
        spark_df = spark_session.createDataFrame(df)

        checker = SparkTypeChecker()
        checker.fit(spark_df)
        checker.transform(spark_df)

    def test_get_pydantic_type(self, spark_session: SparkSession):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "2", "f"]})
        spark_df = spark_session.createDataFrame(df)

        checker = SparkTypeChecker()
        checker.fit(spark_df)

        Model = checker.get_pydantic_type()

        converted = Model.to_model(spark_df)
        assert converted.a == df["a"].to_list()
        assert converted.b == df["b"].to_list()

    @mark.parametrize(["obj"], [[[]], [pd.DataFrame({"a": [1]})]])
    def test_supports_object_fail(self, obj):
        assert not SparkTypeChecker.supports_object(obj)

    def test_supports_object_success(self, spark_session: SparkSession):
        df = spark_session.createDataFrame(pd.DataFrame({"a": [1]}))
        assert SparkTypeChecker.supports_object(df)
