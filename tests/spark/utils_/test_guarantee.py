from pyspark.sql import SparkSession

from mlpype.spark.utils.guarantee import guarantee_spark
from tests.spark.utils import spark_session
from tests.utils import pytest_assert

spark_session


def test_guarantee_spark_with_session(spark_session: SparkSession):
    res = guarantee_spark(spark_session)
    assert res == spark_session


def test_guarantee_spark_with_None(spark_session: SparkSession):
    res = guarantee_spark(None)
    assert res == spark_session


def test_guarantee_spark_error():
    with pytest_assert(AssertionError, "No active SparkSession"):
        guarantee_spark(1)
