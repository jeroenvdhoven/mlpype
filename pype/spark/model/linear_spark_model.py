from typing import Any

from pyspark.ml.regression import LinearRegression

from pype.spark.model.spark_model import SparkModel


class LinearSparkModel(SparkModel[LinearRegression]):
    def _init_model(self, args: dict[str, Any]) -> LinearRegression:
        return LinearRegression(**args)
