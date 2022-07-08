# %%

from pathlib import Path
from typing import Iterable
from unittest.mock import MagicMock

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pype.base.data import DataSetSource
from pype.base.deploy.inference import Inferencer
from pype.base.evaluate.evaluator import Evaluator
from pype.base.experiment.experiment import Experiment
from pype.base.logger.local_logger import LocalLogger
from pype.base.pipeline.pipe import Pipe
from pype.base.pipeline.pipeline import Pipeline
from pype.base.serialiser.joblib_serialiser import JoblibSerialiser
from pype.sklearn.data.data_frame_source import DataFrameSource
from pype.sklearn.model.linear_regression_model import LinearRegressionModel
from pype.sklearn.model.logistic_regression_model import LogisticRegressionModel

# %%

parser = MagicMock()
# parser = ArgumentParser()
model = LinearRegressionModel.get_parameters(parser)

print(parser.add_argument.call_args_list)

# %% [markdown]
# Try a run with sklearn

# %%


def _make_data() -> Iterable[np.ndarray]:
    iris = load_iris()
    x = iris["data"]
    y = iris["target"]

    kept_rows = y < 2
    x = x[kept_rows, :]
    y = y[kept_rows]

    return train_test_split(x, y, test_size=0.2)


# %%
train_x, test_x, train_y, test_y = _make_data()

model = LogisticRegressionModel.from_parameters(
    inputs=["x"],
    outputs=["y"],
)

ds = {
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
        "mse": accuracy_score,
    }
)

pipeline = Pipeline([Pipe("scale", StandardScaler, inputs=["x"], outputs=["x"])])

experiment = Experiment(
    data_sources=ds,
    model=model,
    pipeline=pipeline,
    evaluator=evaluator,
    logger=LocalLogger(),
    serialiser=JoblibSerialiser(),
    output_folder="outputs",
)

metrics = experiment.run()

print(metrics)

# %% [markdown]

# Try loading results again

# %%
folder = Path("outputs")

inferencer = Inferencer.from_folder(folder)

train_x, test_x, train_y, test_y = _make_data()
test_data = DataSetSource(
    x=DataFrameSource(test_x),
    y=DataFrameSource(test_y),
)
result = inferencer.predict(test_data)
print(result)
# %%
