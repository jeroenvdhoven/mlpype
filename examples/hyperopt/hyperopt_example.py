"""Please run this file using `python -m examples.hyperopt.hyperopt_example`.

We do not guarantee results if you use `python examples/hyperopt/hyperopt_example.py`

The goal of this file is to show how to use `mlpype` and `hyperopt` together. The steps are:
1. Create an experiment. For this example, we use the iris dataset and a random forest classifier.
2. Define the hyperopt search space.
3. Use the premade optimise_experiment function to run hyperopt on your experiment.

As per usual, this script ends with loading the model back into memory and running an evaluation.

"""
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# %%
from hyperopt import hp  # type: ignore
from hyperopt.pyll import scope
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
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
from mlpype.hyperopt.optimise import optimise_experiment
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

    return train_test_split(x, y, test_size=0.5)


# %%
train_x, test_x, train_y, test_y = _make_data()

model = SklearnModel.from_sklearn_model_class(RandomForestClassifier, inputs=["x"], outputs=["y"])

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
# %% [markdown]
# Hyperopt configuration
# Here you can set the search space. As you can see, the structure is:
# - Preface all model parameters with model__
# - Preface all pipeline parameters with pipeline__<pipe name>__
#
# This allows you to not just tweak model parameters, but also pipeline params!
# Think of the number of bins in a FeatureHasher.
# Otherwise, this works largely similar to how you'd use hyperopt usually.
search_space = {
    "model__n_estimators": scope.int(hp.quniform("model__n_estimators", 10, 100, 5)),
    "model__max_depth": scope.int(hp.quniform("model__max_depth", 1, 10, 1)),
    "model__min_samples_split": scope.int(hp.quniform("model__min_samples_split", 2, 20, 1)),
    "model__min_samples_leaf": scope.int(hp.quniform("model__min_samples_leaf", 1, 10, 1)),
    "pipeline__scale__with_mean": hp.choice("pipeline__scale__with_mean", [True, False]),
    "pipeline__scale__with_std": hp.choice("pipeline__scale__with_std", [True, False]),
}

_, perf, params, trials = results = optimise_experiment(
    experiment, search_space, max_evals=20, target_metric=("test", "accuracy"), minimise_target=False
)

best_exp = experiment.copy(params)
best_exp.run()

# %%
# Try loading results again
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
