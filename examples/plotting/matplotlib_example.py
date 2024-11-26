"""Please run this file using `python -m examples.plotting.matplotlib_example`.

We do not guarantee results if you use `python examples/plotting/matplotlib_example.py`

This example shows how you can add matplotlib plots to your experiment. These will be automatically
logged. It uses the Plotter class and some custom functions for creating simple charts to make sure
charts can be made for every experiment you run using this setup.
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
from mlpype.base.logger.local_logger import LocalLogger
from mlpype.base.pipeline.pipe import Pipe
from mlpype.base.pipeline.pipeline import Pipeline
from mlpype.base.serialiser.joblib_serialiser import JoblibSerialiser
from mlpype.sklearn.data.data_frame_source import DataFrameSource
from mlpype.sklearn.model import LinearRegressionModel, LogisticRegressionModel
from mlpype.sklearn.pipeline.numpy_type_checker import NumpyTypeChecker
from mlpype.sklearn.pipeline.pandas_type_checker import PandasTypeChecker

# %%


def plot_predictions(
    path: Path,
    y: np.ndarray,
) -> None:
    """Plots histogram of predictions."""
    plt.hist(y, bins=50)
    plt.savefig(path)
    plt.close()


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

pipeline = Pipeline([Pipe("scale", StandardScaler, inputs=["x"], outputs=["x"])])
of = Path("outputs")

pred_ds_name = Constants.PREDICTION_SUFFIX
experiment = Experiment(
    data_sources=ds,
    model=model,
    pipeline=pipeline,
    evaluator=evaluator,
    logger=LocalLogger(),
    type_checker_classes=tcc,
    serialiser=JoblibSerialiser(),
    output_folder=of,
    plots=[
        Plotter(plot_predictions, "predictions.png", [f"y{pred_ds_name}"]),
        Plotter(preds_vs_true, "predictions_vs_true.png", ["y", f"y{pred_ds_name}"]),
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
