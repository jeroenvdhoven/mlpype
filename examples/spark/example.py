"""Please run this file using `python -m examples.spark.example`.

We do not guarantee results if you use `python examples/spark/example.py`

This example shows how to use `mlpype` and `pyspark` together. The steps are:
1. Create an experiment. For this example, we use a dummy dataset and a linear regression classifier.
2. Run the experiment.

This requires that spark is installed on your system. Spark requires a few differen tools, like
- SparkPipe: These are pipe segments more tuned to be used with the pyspark framework, especially for serialisation.
- SparkSerialiser: This is a serialiser for spark pipelines and models.

As per usual, this script ends with loading the model back into memory and running an evaluation.
"""


from pathlib import Path

import pandas as pd
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession

from mlpype.base.data import DataCatalog
from mlpype.base.deploy import Inferencer
from mlpype.base.experiment import Experiment
from mlpype.base.logger import LocalLogger
from mlpype.base.pipeline import Pipeline
from mlpype.base.pipeline.type_checker import TypeCheckerPipe
from mlpype.spark.data.spark_data_frame_source import SparkDataFrameSource
from mlpype.spark.evaluate.spark_evaluator import SparkEvaluator
from mlpype.spark.model import LinearSparkModel
from mlpype.spark.pipeline import SparkTypeChecker
from mlpype.spark.pipeline.spark_pipe import SparkPipe
from mlpype.spark.serialiser.spark_serialiser import SparkSerialiser

ss = SparkSession.builder.getOrCreate()

data = {
    "train": DataCatalog(
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

model = LinearSparkModel(["x"], ["x"], featuresCol="a_scaled", labelCol="target", predictionCol="prediction")

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


# Test loading again from folder.
inf_path = Inferencer.from_folder(Path("outputs"))
prediction = inf_path.predict(data["train"])

print(prediction["x"].toPandas())
