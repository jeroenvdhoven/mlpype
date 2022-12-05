from typing import Any

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession

from pype.base.data.data_source import DataSource


class SparkReadSource(DataSource[SparkDataFrame]):
    def __init__(
        self,
        spark_session: SparkSession,
        file: str,
        format: str,
        options: dict[str, Any],
    ) -> None:
        super().__init__()
        self.spark_session = spark_session
        self.file = file
        self.format = format
        self.options = options

    def read(self) -> SparkDataFrame:
        return self.spark_session.read.format(self.format).load(self.file, **self.options)
