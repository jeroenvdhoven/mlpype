from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pandas.testing import assert_frame_equal
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession

from mlpype.base.data import DataSet
from mlpype.base.pipeline import Pipeline
from mlpype.spark.pipeline import SparkPipe
from mlpype.spark.serialiser import SparkSerialiser
from tests.spark.utils import spark_session

spark_session


class Test_SparkSerialiser:
    def test_serialise_pipeline(self, spark_session: SparkSession):
        data = DataSet(
            x=spark_session.createDataFrame(pd.DataFrame({"a": [1, 2, 3.0], "target": [2, 3, 4.0]})),
            helper=spark_session.createDataFrame(pd.DataFrame({"a": [2, 3, 4]})),
        )

        pipeline = Pipeline(
            [
                SparkPipe(
                    "assemble", VectorAssembler, ["x"], ["x"], kw_args={"inputCols": ["a"], "outputCol": "a_gathered"}
                ),
                SparkPipe(
                    "scale", StandardScaler, ["x"], ["x"], kw_args={"inputCol": "a_gathered", "outputCol": "a_scaled"}
                ),
            ]
        )

        pipeline.fit(data)

        serialiser = SparkSerialiser()

        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            file = tmp_dir / "tmp"
            serialiser.serialise(pipeline, file)

            loaded = serialiser.deserialise(file)

            old_data = pipeline.transform(data)
            new_data = loaded.transform(data)

            assert len(old_data) == len(new_data)
            for name, old_value in old_data.items():
                new_value = new_data[name]
                assert_frame_equal(old_value.toPandas(), new_value.toPandas())

    def test_serialise_joblib(self):
        serialiser = SparkSerialiser()

        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            file = tmp_dir / "tmp"
            a = Path("a") / "b"

            serialiser.serialise(a, file)
            loaded = serialiser.deserialise(file)

            assert a == loaded
