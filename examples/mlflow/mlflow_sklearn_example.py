"""Please run this file using `python -m examples.mlflow.mlflow_sklearn_example --pipeline__impute__verbose 0`.

We do not guarantee results if you use `python examples/mlflow/mlflow_sklearn_example.py`
You can use command line arguments in this example, such as:
python -m examples.mlflow.mlflow_sklearn_example --model__fit_intercept=False --pipeline__impute__verbose 0

This requires the pype.sklearn package to also be installed.

Please make sure mlflow is running locally before starting this script, e.g. by using `mlflow ui`
If you are running this example locally, make sure you run this from the top level directory.
This will make sure the artifacts will show up.
"""
# %%

from pathlib import Path
from typing import Iterable

import numpy as np
from mlflow.tracking.fluent import get_experiment_by_name
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pype.base.data import DataSetSource
from pype.base.deploy.inference import Inferencer
from pype.base.evaluate.evaluator import Evaluator
from pype.base.experiment.experiment import Experiment
from pype.base.pipeline.pipe import Pipe
from pype.base.pipeline.pipeline import Pipeline
from pype.base.pipeline.type_checker import TypeCheckerPipe
from pype.base.serialiser.joblib_serialiser import JoblibSerialiser
from pype.mlflow.logger.mlflow_logger import MlflowLogger
from pype.sklearn.data.data_frame_source import DataFrameSource
from pype.sklearn.model.logistic_regression_model import LogisticRegressionModel
from pype.sklearn.pipeline.numpy_type_checker import NumpyTypeChecker
from pype.sklearn.pipeline.pandas_type_checker import PandasTypeChecker

# %% [markdown]
# Try a run with sklearn

# %%
# Make a logger using Mlflow. Ensure it starts with 'http://', or you will get connection issues.
# Make sure you start mlflow before running this!
experiment_name = "jeroen-example-experiment"
logger = MlflowLogger(experiment_name, "http://localhost:5000")

# for on-databricks logging. This will also log artifacts.
# experiment_name = "/Users/<user name>/<experiment name>"
# logger = MlflowLogger(experiment_name, "databricks")

# %%
#  Try a run with sklearn and argument reading


def _make_data() -> Iterable[np.ndarray]:
    iris = load_iris()
    x = iris["data"]
    y = iris["target"]

    kept_rows = y < 2
    x = x[kept_rows, :]
    y = y[kept_rows]

    return train_test_split(x, y, test_size=0.2)


train_x, test_x, train_y, test_y = _make_data()

datasets = {
    "train": DataSetSource(
        x=DataFrameSource(train_x),
        y=DataFrameSource(train_y),
    ),
    "test": DataSetSource(
        x=DataFrameSource(test_x),
        y=DataFrameSource(test_y),
    ),
}

evaluator = Evaluator(
    {
        "accuracy": accuracy_score,
    }
)

tcc = [NumpyTypeChecker, PandasTypeChecker]

input_ds_type_checker = TypeCheckerPipe(
    "type_checker-in",
    input_names=["x"],
    type_checker_classes=tcc,
)

output_ds_type_checker = TypeCheckerPipe(
    "type_checker-out",
    input_names=["y"],
    type_checker_classes=tcc,
)

pipeline = Pipeline(
    [
        Pipe("scale", StandardScaler, inputs=["x"], outputs=["x"]),
        Pipe("impute", SimpleImputer, inputs=["x"], outputs=["x"], kw_args={"missing_values": ""}),
    ]
)
of = Path("outputs")

experiment = Experiment.from_command_line(
    data_sources=datasets,
    model_class=LogisticRegressionModel,
    model_inputs=["x"],
    model_outputs=["y"],
    pipeline=pipeline,
    evaluator=evaluator,
    logger=logger,
    input_type_checker=input_ds_type_checker,
    output_type_checker=output_ds_type_checker,
    serialiser=JoblibSerialiser(),
    output_folder=of,
)

metrics = experiment.run()
print("Metrics:", metrics)

# Try loading results again
folder = Path("outputs")
inferencer = Inferencer.from_folder(folder)

train_x, test_x, train_y, test_y = _make_data()
test_data = DataSetSource(
    x=DataFrameSource(test_x),
)
result = inferencer.predict(test_data)
print(result)

# %%
# Get artifact path from mlflow

exp = get_experiment_by_name(experiment_name)
print("artifact_location:", exp.artifact_location)
