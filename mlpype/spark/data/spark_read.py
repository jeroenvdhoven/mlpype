from typing import Any, Dict

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession

from mlpype.base.data.data_source import DataSource


class SparkReadSource(DataSource[SparkDataFrame]):
    def __init__(
        self,
        spark_session: SparkSession,
        file: str,
        format: str,
        options: Dict[str, Any],
    ) -> None:
        """Read a file through Spark.

        Args:
            spark_session (SparkSession): The active SparkSessionl.
            file (str): The file path.
            format (str): The file format.
            options (Dict[str, Any]): Any additional options for `load`.
        """
        super().__init__()
        self.spark_session = spark_session
        self.file = file
        self.format = format
        self.options = options

    def read(self) -> SparkDataFrame:
        """Read the file at the given location.

        Returns:
            SparkDataFrame: A DataFrame based on the given file.
        """
        return self.spark_session.read.format(self.format).load(self.file, **self.options)
