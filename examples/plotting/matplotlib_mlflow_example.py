"""Please run this file using `python -m examples.plotting.matplotlib_mlflow_example`.

We do not guarantee results if you use `python examples/plotting/matplotlib_mlflow_example.py`

This file shows how to combine plots and MLflow to log created charts directly to MLflow. It includes:
- Examples using MatplotlibPlotter for plots specifically using matplotlib.
- Examples using ShapleyPlot for auto generating shapley plots.
- Examples for the base Plotter.

Please make sure mlflow is running locally before starting this script, e.g. by using `mlflow ui`
If you are running this example locally, make sure you run this from the top level directory.
This will make sure the artifacts will show up.
"""
# %%

from pathlib import Path
from typing import Iterable
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlpype.base.constants import Constants
from mlpype.base.data import DataCatalog
from mlpype.base.deploy.inference import Inferencer
from mlpype.base.evaluate.evaluator import Evaluator
from mlpype.base.evaluate.plot import Plotter
from mlpype.base.experiment.experiment import Experiment
from mlpype.base.pipeline.pipe import Pipe
from mlpype.base.pipeline.pipeline import Pipeline
from mlpype.base.serialiser.joblib_serialiser import JoblibSerialiser
from mlpype.matplotlib.evaluate import MatplotlibPlotter, ShapleyPlot
from mlpype.mlflow.logger.mlflow_logger import MlflowLogger
from mlpype.sklearn.data.data_frame_source import DataFrameSource
from mlpype.sklearn.model import LinearRegressionModel, LogisticRegressionModel
from mlpype.sklearn.pipeline.numpy_type_checker import NumpyTypeChecker
from mlpype.sklearn.pipeline.pandas_type_checker import PandasTypeChecker

experiment_name = "plot-example-experiment"
logger = MlflowLogger(experiment_name, "http://127.0.0.1:5000")


def plot_predictions(
    y: np.ndarray,
) -> None:
    """Plots histogram of predictions."""
    plt.hist(y, bins=50)


def preds_vs_true(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Plots predictions vs true."""
    plt.scatter(y_true, y_pred)
    plt.xlabel("Labels")
    plt.ylabel("Predictions")
    plt.savefig(path)
    plt.close()


# %%

parser = MagicMock()
model = LinearRegressionModel.get_parameters(parser)

# %% [markdown]
# Try a run with sklearn

# %%


def _make_data() -> Iterable[np.ndarray]:
    iris = load_iris(as_frame=True)
    x = pd.DataFrame(iris["data"])
    y = pd.DataFrame(iris["target"])

    kept_rows = y["target"] < 2
    x = x.loc[kept_rows, :]
    y = y.loc[kept_rows, :]

    return train_test_split(x, y, test_size=0.2)


# %%
train_x, test_x, train_y, test_y = _make_data()

model = LogisticRegressionModel(
    model=None,
    inputs=["x"],
    outputs=["y"],
)

ds = {
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


class StandardDfScaler(StandardScaler):
    """Standard scaler for pandas DFs."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the given DF but keep the DF class."""
        res = super().transform(df)
        return pd.DataFrame(res, index=df.index, columns=df.columns)


pipeline = Pipeline([Pipe("scale", StandardDfScaler, inputs=["x"], outputs=["x"])])

of = Path("outputs")

pred_ds_name = Constants.PREDICTION_SUFFIX
experiment = Experiment(
    data_sources=ds,
    model=model,
    pipeline=pipeline,
    evaluator=evaluator,
    logger=logger,
    type_checker_classes=tcc,
    serialiser=JoblibSerialiser(),
    output_folder=of,
    plots=[
        MatplotlibPlotter(plot_predictions, "predictions.png", [f"y{pred_ds_name}"]),
        Plotter(preds_vs_true, "predictions_vs_true.png", ["y", f"y{pred_ds_name}"]),
        ShapleyPlot("x", "y", sample_size=30),
    ],
)

metrics = experiment.run()

print(metrics)

# %% [markdown]

# Try loading results again

# %%
folder = Path("outputs")

inferencer = Inferencer.from_folder(folder)

train_x, test_x, train_y, test_y = _make_data()
test_data = DataCatalog(
    x=DataFrameSource(test_x),
    y=DataFrameSource(test_y),
)
result = inferencer.predict(test_data)
print(result)
# %%
