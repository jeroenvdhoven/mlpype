"""Call this using some form of command line.

e.g. python -m examples.sklearn.sklearn_example_from_cmd_args --model__fit_intercept=False

The goal of this file is to show how to use `mlpype` and `sklearn` together.
It also shows how you can pass arguments using the command line. The steps are:
1. Create an experiment using from_command_line. For this example, we use the iris dataset and a
    logistic regression classifier.
    a. Show how to use SklearnModel.from_sklearn_model_class to reuse Sklearn models.
2. Run the experiment.

Like arguments provided in hyperopt, you can also provide arguments using the command line.
    They follow the same convention:
    - For model arguments, preface the argument name with `model__`.
    - For pipeline arguments, preface the argument name with `pipeline__<pipe name>__`.
        This uses the argument for the given pipe.

As per usual, this script ends with loading the model back into memory and running an evaluation.
"""

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
from mlpype.base.pipeline.type_checker import TypeCheckerPipe
from mlpype.base.serialiser.joblib_serialiser import JoblibSerialiser
from mlpype.sklearn.data.data_frame_source import DataFrameSource
from mlpype.sklearn.model.sklearn_model import SklearnModel
from mlpype.sklearn.pipeline.numpy_type_checker import NumpyTypeChecker
from mlpype.sklearn.pipeline.pandas_type_checker import PandasTypeChecker


#  Try a run with sklearn and argument reading
def _make_data() -> Iterable[np.ndarray]:
    iris = load_iris(as_frame=True)
    x = pd.DataFrame(iris["data"])
    y = pd.DataFrame(iris["target"])

    kept_rows = y["target"] < 2
    x = x.loc[kept_rows, :]
    y = y.loc[kept_rows, :]

    return train_test_split(x, y, test_size=0.2)


train_x, test_x, train_y, test_y = _make_data()

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

pipeline = Pipeline([Pipe("scale", StandardScaler, inputs=["x"], outputs=["x"])])
of = Path("outputs")

experiment = Experiment.from_command_line(
    data_sources=ds,
    model_class=SklearnModel.class_from_sklearn_model_class(LogisticRegression),
    model_inputs=["x"],
    model_outputs=["y"],
    pipeline=pipeline,
    evaluator=evaluator,
    logger=LocalLogger(),
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
test_data = DataCatalog(
    x=DataFrameSource(test_x),
    y=DataFrameSource(test_y),
)
result = inferencer.predict(test_data)
print(result)
