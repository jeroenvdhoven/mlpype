from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession

from pype.base.data.data_source import DataSource


class SparkSqlSource(DataSource[SparkDataFrame]):
    def __init__(
        self,
        spark_session: SparkSession,
        query: str,
    ) -> None:
        super().__init__()
        self.query = query
        self.spark_session = spark_session

    def read(self) -> SparkDataFrame:
        return self.spark_session.sql(self.query)
