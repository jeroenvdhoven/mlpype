"""Please run this file using `python -m examples.tensorflow.keras_model_example`.

We do not guarantee results if you use `python examples/tensorflow/keras_model_example.py`

The goal of this file is to show how to use `mlpype` and `tensorflow` together. The steps are:
1. Create an experiment. For this example, we use the iris dataset and a feedforward MLP keras model.
2. Run the experiment.
"""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
from keras.optimizers import Nadam
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlpype.base.data import DataCatalog
from mlpype.base.deploy.inference import Inferencer
from mlpype.base.evaluate.evaluator import Evaluator
from mlpype.base.experiment.experiment import Experiment
from mlpype.base.logger.local_logger import LocalLogger
from mlpype.base.pipeline.operator import Operator
from mlpype.base.pipeline.pipe import Pipe
from mlpype.base.pipeline.pipeline import Pipeline
from mlpype.base.pipeline.type_checker import TypeCheckerPipe
from mlpype.base.serialiser.joblib_serialiser import JoblibSerialiser
from mlpype.sklearn.data.data_frame_source import DataFrameSource
from mlpype.sklearn.pipeline.numpy_type_checker import NumpyTypeChecker
from mlpype.sklearn.pipeline.pandas_type_checker import PandasTypeChecker
from mlpype.tensorflow.model import MLPPypeModel
from mlpype.tensorflow.pipeline.tensor_checker import TensorflowTypeChecker


class NumpyToTensor(Operator):
    """Convert the numpy array into a Tensor."""

    def fit(self, *_: np.ndarray) -> "Operator":
        """Skipped."""
        return self

    def transform(self, data: np.ndarray) -> tf.Tensor:  # type: ignore
        """Transform the numpy array into a Tensor."""
        return tf.convert_to_tensor(data)  # type: ignore


if __name__ == "__main__":
    tcc = [NumpyTypeChecker, PandasTypeChecker, TensorflowTypeChecker]  # type: ignore

    def _make_data() -> Iterable[np.ndarray]:
        iris = load_iris(as_frame=True)
        x = pd.DataFrame(iris["data"])
        y = pd.DataFrame(iris["target"])

        kept_rows = y["target"] < 2
        x = x.loc[kept_rows, :]
        y = y.loc[kept_rows, :]

        return train_test_split(x, y, test_size=0.2)

    train_x, test_x, train_y, test_y = _make_data()

    model = MLPPypeModel(
        model=None,
        inputs=["x_tf"],
        outputs=["y_tf"],
        loss=BinaryCrossentropy(),
        optimizer_class=Nadam,
        metrics=[BinaryAccuracy()],
        epochs=10,
        batch_size=32,
        learning_rate=0.003,
        output_size=1,
        n_layers=2,
        layer_size=8,
        activation="selu",
        output_activation="sigmoid",
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
            "accuracy": BinaryAccuracy(),
        }
    )

    input_ds_type_checker = TypeCheckerPipe(
        "type_checker-in",
        input_names=["x"],
        type_checker_classes=tcc,
    )

    output_ds_type_checker = TypeCheckerPipe(
        "type_checker-out",
        input_names=["y_tf"],
        type_checker_classes=tcc,
    )

    pipeline = Pipeline(
        [
            Pipe("scale", StandardScaler, inputs=["x"], outputs=["x_scaled"]),
            Pipe("to_tf_x", NumpyToTensor, inputs=["x_scaled"], outputs=["x_tf"]),
            Pipe("to_tf_y", NumpyToTensor, inputs=["y"], outputs=["y_tf"], skip_on_inference=True),
        ]
    )
    of = Path("outputs")

    this_file = Path(__file__)

    experiment = Experiment(
        data_sources=ds,
        model=model,
        pipeline=pipeline,
        evaluator=evaluator,
        logger=LocalLogger(),
        input_type_checker=input_ds_type_checker,
        output_type_checker=output_ds_type_checker,
        serialiser=JoblibSerialiser(),
        output_folder=of,
        # Need to add this file to output to make sure we can import CustomModel and CustomStandardScaler/tcc
        additional_files_to_store=[this_file],
    )

    metrics = experiment.run()

    print(metrics)

    # Try loading results again

    folder = Path("outputs")

    inferencer = Inferencer.from_folder(folder)

    train_x, test_x, train_y, test_y = _make_data()
    test_data = DataCatalog(
        x=DataFrameSource(test_x),
    )
    result = inferencer.predict(test_data)
    print(result)
