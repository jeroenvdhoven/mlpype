from pyspark.sql import DataFrame as SparkDataFrame

from mlpype.base.data.data_source import DataSource


class SparkDataFrameSource(DataSource[SparkDataFrame]):
    # should only be used for testing purposes!
    def __init__(
        self,
        df: SparkDataFrame,
    ) -> None:
        """Returns a fixed spark dataframe reference.

        This should only be used for testing/debugging purposes.
        Consider the SQL or Read classes for proper data reading!

        Args:
            df (SparkDataFrame): The DataFrame to use as output when
                read() is called.
        """
        super().__init__()
        self.df = df

    def read(self) -> SparkDataFrame:
        """Returns the stored DataFrame.

        Returns:
            SparkDataFrame: The stored DataFrame.
        """
        return self.df
