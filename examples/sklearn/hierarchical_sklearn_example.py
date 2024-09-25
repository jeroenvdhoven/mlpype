"""Please run this file using `python -m examples.sklearn.hierarchical_sklearn_example`.

We do not guarantee results if you use `python examples/sklearn/hierarchical_sklearn_example.py`
"""
# %%

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlpype.base.data import DataCatalog
from mlpype.base.deploy.inference import Inferencer
from mlpype.base.evaluate.evaluator import Evaluator
from mlpype.base.experiment.experiment import Experiment
from mlpype.base.logger.local_logger import LocalLogger
from mlpype.base.model.hierarchical_model import HierarchicalModel
from mlpype.base.pipeline.pipe import Pipe
from mlpype.base.pipeline.pipeline import Pipeline
from mlpype.base.serialiser.joblib_serialiser import JoblibSerialiser
from mlpype.sklearn.data.data_frame_source import DataFrameSource
from mlpype.sklearn.model import LogisticRegressionModel
from mlpype.sklearn.pipeline.numpy_type_checker import NumpyTypeChecker
from mlpype.sklearn.pipeline.pandas_type_checker import PandasTypeChecker

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
# Data splitters and mergers


def split_data(array: np.ndarray) -> Dict[str, np.ndarray]:
    """Split a numpy array in half.

    Not the most useful, but one can imagine splitting by a key on a DataFrame in a RL example
    """
    n = int(array.shape[0] / 2)
    return {
        "lower": array[:n],
        "upper": array[n:],
    }


def merge_data(arrays: Dict[str, np.ndarray]) -> np.ndarray:
    """Merge all numpy arrays in the dictionary."""
    return np.concatenate([*arrays.values()], axis=0)


# %%
train_x, test_x, train_y, test_y = _make_data()


class HierLogisticRegressionModel(HierarchicalModel[LogisticRegressionModel]):
    """A Hierarchical model for Logistic Regressino submodels."""


model = HierLogisticRegressionModel(
    inputs=["x"],
    outputs=["y"],
    data_splitter=split_data,
    data_merger=merge_data,
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

experiment = Experiment(
    data_sources=ds,
    model=model,
    pipeline=pipeline,
    evaluator=evaluator,
    logger=LocalLogger(),
    type_checker_classes=tcc,
    serialiser=JoblibSerialiser(),
    output_folder=of,
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
