"""Please run this file using `python -m examples.sklearn.sklearn_example`.

We do not guarantee results if you use `python examples/sklearn/sklearn_example.py`

The goal of this file is to show how to use `mlpype` and `sklearn` together. The steps are:
1. Show how to use SklearnModel.from_sklearn_model_class to reuse Sklearn models.
2. Create an experiment. For this example, we use the iris dataset and a logistic regression classifier.
3. Run the experiment.

As per usual, this script ends with loading the model back into memory and running an evaluation.

"""
# %%

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlpype.base.data import DataCatalog
from mlpype.base.deploy.inference import Inferencer
from mlpype.base.evaluate.evaluator import Evaluator
from mlpype.base.experiment.experiment import Experiment
from mlpype.base.logger.local_logger import LocalLogger
from mlpype.base.pipeline.pipe import Pipe
from mlpype.base.pipeline.pipeline import Pipeline
from mlpype.base.serialiser.joblib_serialiser import JoblibSerialiser
from mlpype.sklearn.data.data_frame_source import DataFrameSource
from mlpype.sklearn.model import SklearnModel
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
train_x, test_x, train_y, test_y = _make_data()

model = SklearnModel.from_sklearn_model_class(LogisticRegression, inputs=["x"], outputs=["y"])
# model = LogisticRegressionModel(
#     model=None,
#     inputs=["x"],
#     outputs=["y"],
# )

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
