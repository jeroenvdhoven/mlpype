"""Please run this file using `python -m examples.tensorflow.keras_model_example`.

We do not guarantee results if you use `python examples/tensorflow/keras_model_example.py`
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

from pype.base.data import DataSetSource
from pype.base.deploy.inference import Inferencer
from pype.base.evaluate.evaluator import Evaluator
from pype.base.experiment.experiment import Experiment
from pype.base.logger.local_logger import LocalLogger
from pype.base.pipeline.operator import Operator
from pype.base.pipeline.pipe import Pipe
from pype.base.pipeline.pipeline import Pipeline
from pype.base.pipeline.type_checker import TypeCheckerPipe
from pype.base.serialiser.joblib_serialiser import JoblibSerialiser
from pype.sklearn.data.data_frame_source import DataFrameSource
from pype.sklearn.pipeline.numpy_type_checker import NumpyTypeChecker
from pype.sklearn.pipeline.pandas_type_checker import PandasTypeChecker
from pype.tensorflow.model import MLPPypeModel
from pype.tensorflow.pipeline.tensor_checker import TensorflowTypeChecker


class NumpyToTensor(Operator):
    def fit(self, *_: np.ndarray) -> "Operator":
        """Skipped."""
        return self

    def transform(self, data: np.ndarray) -> tf.Tensor:  # type: ignore
        """Transform the numpy array into a Tensor."""
        return tf.convert_to_tensor(data)  # type: ignore


if __name__ == "__main__":
    tcc = [
        (np.ndarray, NumpyTypeChecker),
        (pd.DataFrame, PandasTypeChecker),
        (tf.Tensor, TensorflowTypeChecker),  # type: ignore
    ]

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
    test_data = DataSetSource(
        x=DataFrameSource(test_x),
    )
    result = inferencer.predict(test_data)
    print(result)
