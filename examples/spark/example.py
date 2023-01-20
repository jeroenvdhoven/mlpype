"""Please run this file using `python -m examples.spark.example`.

We do not guarantee results if you use `python examples/spark/example.py`
"""


import pandas as pd
from pipeline.type_checker import TypeCheckerPipe
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession

from pype.base.data import DataSetSource
from pype.base.deploy import Inferencer
from pype.base.experiment import Experiment
from pype.base.logger import LocalLogger
from pype.base.pipeline import Pipeline
from pype.spark.data.spark_data_frame_source import SparkDataFrameSource
from pype.spark.evaluate.spark_evaluator import SparkEvaluator
from pype.spark.model import LinearSparkModel
from pype.spark.pipeline import SparkTypeChecker
from pype.spark.pipeline.spark_pipe import SparkPipe
from pype.spark.serialisation.spark_serialiser import SparkSerialiser

ss = SparkSession.builder.getOrCreate()

data = {
    "train": DataSetSource(
        x=SparkDataFrameSource(ss.createDataFrame(pd.DataFrame({"a": [1, 2, 3.0], "target": [2, 3, 4.0]}))),
        helper=SparkDataFrameSource(ss.createDataFrame(pd.DataFrame({"a": [2, 3, 4]}))),
    )
}

pipeline = Pipeline(
    [
        SparkPipe("assemble", VectorAssembler, ["x"], ["x"], kw_args={"inputCols": ["a"], "outputCol": "a_gathered"}),
        SparkPipe("scale", StandardScaler, ["x"], ["x"], kw_args={"inputCol": "a_gathered", "outputCol": "a_scaled"}),
    ]
)

model = LinearSparkModel(
    ["x"], ["x"], featuresCol="a_scaled", labelCol="target", predictionCol="prediction", output_col="prediction"
)

logger = LocalLogger()

evaluator = SparkEvaluator(
    BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="target"),
    metrics=["areaUnderROC", "areaUnderPR"],
)

experiment = Experiment(
    data_sources=data,
    model=model,
    pipeline=pipeline,
    evaluator=evaluator,
    logger=logger,
    serialiser=SparkSerialiser(),
    output_folder="outputs",
    input_type_checker=TypeCheckerPipe("inputs", ["x"], [SparkTypeChecker]),
    output_type_checker=TypeCheckerPipe("outputs", ["x"], [SparkTypeChecker]),
)

experiment.run()

# Test running
inferencer = Inferencer.from_experiment(experiment)
prediction = inferencer.predict(data["train"])

print(prediction["x"].toPandas())
