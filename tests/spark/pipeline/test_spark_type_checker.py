from datetime import date
from typing import Dict, List, Union
from unittest.mock import MagicMock, call, patch

import pandas as pd
from pandas.testing import assert_frame_equal
from pydantic import create_model
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    NumericType,
    StringType,
    StructField,
    StructType,
)
from pytest import mark

from mlpype.spark.pipeline.spark_type_checker import SparkData, SparkTypeChecker
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
        dtypes = MagicMock()
        data.schema = dtypes
        checker = SparkTypeChecker()

        with patch.object(checker, "_convert_dtypes") as mock_convert:
            result = checker.fit(data)

        mock_convert.assert_called_once_with(dtypes)
        assert result == checker

    def test_convert_dtypes(self):
        checker = SparkTypeChecker()
        schema = MagicMock()
        schema.fields = StructType([StructField("colA", IntegerType()), StructField("colB", StringType())])
        returns = [2, 3]

        with patch.object(checker, "_convert_dtype", side_effect=returns) as mock_convert:
            result = checker._convert_dtypes(schema)

        mock_convert.assert_has_calls([call(f) for f in schema.fields])

        assert result == {"colA": 2, "colB": 3}

    @mark.parametrize(
        ["name", "field", "expected"],
        [
            ["String", StructField("String", StringType()), (StringType, str)],
            ["Integer", StructField("Integer", IntegerType()), (IntegerType, int)],
            ["Numeric", StructField("Numeric", NumericType()), (NumericType, float)],
            ["Double", StructField("Double", DoubleType()), (NumericType, float)],
            ["Float", StructField("Float", FloatType()), (NumericType, float)],
            ["Boolean", StructField("Boolean", BooleanType()), (BooleanType, bool)],
            ["Date", StructField("Date", DateType()), (DateType, date)],
            ["Vector", StructField("Vector", VectorUDT()), (VectorUDT, list[float])],
        ],
    )
    def test_convert_dtype(self, name: str, field: StructField, expected: type):
        checker = SparkTypeChecker()
        result = checker._convert_dtype(field)

        assert result == expected

    def test_convert_dtype_assertion(self):
        checker = SparkTypeChecker()
        with pytest_assert(ValueError, "ArrayType(IntegerType(), True) not supported"):
            t = StructField("failure", ArrayType(IntegerType()))
            checker._convert_dtype(t)

    @mark.parametrize(
        ["name", "error_on_missing_column"],
        [
            ["Should error", True],
            ["Should not error", False],
        ],
    )
    def test_transform_assert_missing_columns(
        self, spark_session: SparkSession, name: str, error_on_missing_column: bool
    ):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "2", "f"]})
        spark_df = spark_session.createDataFrame(df)

        missing_df = pd.DataFrame({"a": [1, 2, 3]})
        spark_missing_df = spark_session.createDataFrame(missing_df)

        checker = SparkTypeChecker(error_on_missing_column=error_on_missing_column)
        checker.fit(spark_df)

        if error_on_missing_column:
            with pytest_assert(AssertionError, "`b` is missing from the dataset"):
                checker.transform(spark_missing_df)
        else:
            # no error thrown
            checker.transform(spark_missing_df)

    def test_transform_check_dtype(self, spark_session: SparkSession):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "2", "f"]})
        spark_df = spark_session.createDataFrame(df)

        missing_df = pd.DataFrame({"a": [1, 2, 3], "b": [2.2, 3.4, 5.6]})
        spark_missing_df = spark_session.createDataFrame(missing_df)

        checker = SparkTypeChecker()
        checker.fit(spark_df)

        with pytest_assert(AssertionError, "Dtypes did not match up for col b: Expected StringType, got DoubleType"):
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

        name = "spark_data"
        checker = SparkTypeChecker(name)
        checker.fit(spark_df)

        Model = checker.get_pydantic_type()
        assert Model.__name__ == f"SparkData[{name}]"

        converted = Model.to_model(spark_df)
        assert converted.a == df["a"].to_list()
        assert converted.b == df["b"].to_list()

    @mark.parametrize(["obj"], [[[]], [pd.DataFrame({"a": [1]})]])
    def test_supports_object_fail(self, obj):
        assert not SparkTypeChecker.supports_object(obj)

    def test_supports_object_success(self, spark_session: SparkSession):
        df = spark_session.createDataFrame(pd.DataFrame({"a": [1]}))
        assert SparkTypeChecker.supports_object(df)
