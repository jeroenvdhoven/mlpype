"""A mlpype-compliant framework for using a Linear Spark Model."""
from typing import Any, Dict

from pyspark.ml.regression import LinearRegression

from mlpype.spark.model.spark_model import SparkModel


class LinearSparkModel(SparkModel[LinearRegression]):
    """A mlpype-compliant framework for using a Linear Spark Model."""

    def _init_model(self, args: Dict[str, Any]) -> LinearRegression:
        return LinearRegression(**args)
