"""Provides tools for using files as an input data source in Spark."""
from typing import Any, Dict, Optional

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession

from mlpype.base.data.data_source import DataSource
from mlpype.spark.utils.guarantee import guarantee_spark


class SparkReadSource(DataSource[SparkDataFrame]):
    """Read a file through Spark."""

    def __init__(
        self,
        file: str,
        format: str,
        options: Dict[str, Any],
        spark_session: Optional[SparkSession] = None,
    ) -> None:
        """Read a file through Spark.

        Args:
            file (str): The file path.
            format (str): The file format.
            options (Dict[str, Any]): Any additional options for `load`.
            spark_session (Optional[SparkSession]): The current SparkSession.
        """
        super().__init__()
        self.spark_session = guarantee_spark(spark_session)
        self.file = file
        self.format = format
        self.options = options

    def read(self) -> SparkDataFrame:
        """Read the file at the given location.

        Returns:
            SparkDataFrame: A DataFrame based on the given file.
        """
        return self.spark_session.read.format(self.format).load(self.file, **self.options)
