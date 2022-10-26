from pyspark.sql import DataFrame as SparkDataFrame

from pype.base.data.data_source import DataSource


class SparkDataFrameSource(DataSource[SparkDataFrame]):
    # should only be used for testing purposes!
    def __init__(
        self,
        df: SparkDataFrame,
    ) -> None:
        super().__init__()
        self.df = df

    def read(self) -> SparkDataFrame:
        return self.df
