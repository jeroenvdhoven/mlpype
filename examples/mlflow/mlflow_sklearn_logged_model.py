"""Please run this file using `python -m examples.mlflow.mlflow_sklearn_logged_model`.

We do not guarantee results if you use `python examples/mlflow/mlflow_sklearn_example.py`

This requires the mlpype.sklearn package to also be installed.

The goal of this file is to show how to use `mlpype` and `mlflow` together, and register your model.
This file will use the MlflowLogger to automatically log your entire experiment to the configured
mlflow server. This example uses a local mlflow server for this purpose. It will show you
how to register a model automatically and use this to load a model back into memory using mlflow's tools.

Please make sure mlflow is running locally before starting this script. The following should work:
`mlflow server --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root ./mlruns`
If you are running this example locally, make sure you run this from the top level directory.
This will make sure the artifacts will show up.
"""
# %%

from pathlib import Path
from typing import Iterable

import numpy as np
from mlflow import search_runs  # type: ignore
from mlflow.pyfunc import load_model
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlpype.base.data import DataCatalog
from mlpype.base.deploy.inference import Inferencer
from mlpype.base.evaluate.evaluator import Evaluator
from mlpype.base.experiment.experiment import Experiment
from mlpype.base.pipeline.pipe import Pipe
from mlpype.base.pipeline.pipeline import Pipeline
from mlpype.base.pipeline.type_checker import TypeCheckerPipe
from mlpype.base.serialiser.joblib_serialiser import JoblibSerialiser
from mlpype.mlflow.logger.mlflow_logger import MlflowLogger
from mlpype.sklearn.data.data_frame_source import DataFrameSource
from mlpype.sklearn.model import LogisticRegressionModel
from mlpype.sklearn.pipeline.numpy_type_checker import NumpyTypeChecker
from mlpype.sklearn.pipeline.pandas_type_checker import PandasTypeChecker

# %% [markdown]
# Try a run with sklearn

# %%
# Make a logger using Mlflow. Ensure it starts with 'http://', or you will get connection issues.
# Make sure you start mlflow before running this!
experiment_name = "example-experiment"
model_name = "example-model"
logger = MlflowLogger(
    experiment_name,
    "http://127.0.0.1:5000",
    artifact_location="mlruns",
    # Setting the model name is optional, but will automatically register your model.
    model_name=model_name,
)

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
    "train": DataCatalog(
        x=DataFrameSource(train_x),
        y=DataFrameSource(train_y),
    ),
    "test": DataCatalog(
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

experiment = Experiment.from_dictionary(
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
    parameters={
        "model__fit_intercept": False,
        "pipeline__impute__verbose": 0,
    },
)

metrics = experiment.run()
print("Metrics:", metrics)

# %%
# Try loading results again
folder = Path("outputs")
inferencer = Inferencer.from_folder(folder)

train_x, test_x, train_y, test_y = _make_data()
test_data = DataCatalog(
    x=DataFrameSource(test_x),
)
result = inferencer.predict(test_data)
print(result)

# %%
# If you set model_name, it will automatically register the model. Otherwise, you can manually do this.
# # Log model in MLflow
# exp = get_experiment_by_name(experiment_name)
# # Get the last run in this experiment.
# runs = search_runs(experiment_ids=exp.experiment_id, output_format="list")
# assert len(runs) > 0
# run = runs[0]
# logger.register_mlpype_model(run.info.run_id, "mlflow_testing_model")

# %%
model = load_model("models:/mlflow_testing_model/latest")
result = model.predict(test_data)
print("Results from py-func loaded model: ", str(result))
