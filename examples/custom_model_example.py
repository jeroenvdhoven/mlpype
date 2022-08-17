"""Please run this file using `python -m examples.custom_model_example`.

We do not guarantee results if you use `python examples/custom_model_example.py`
"""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from pype.base.data import DataSetSource
from pype.base.deploy.inference import Inferencer
from pype.base.evaluate.evaluator import Evaluator
from pype.base.experiment.experiment import Experiment
from pype.base.logger.local_logger import LocalLogger
from pype.base.pipeline.pipe import Pipe
from pype.base.pipeline.pipeline import Pipeline
from pype.base.pipeline.type_checker import TypeCheckerPipe
from pype.base.serialiser.joblib_serialiser import JoblibSerialiser
from pype.sklearn.data.data_frame_source import DataFrameSource
from pype.sklearn.model.logistic_regression_model import LogisticRegressionModel

from .custom_processer import CustomStandardScaler, tcc


# classes / functions defined in the core file can be imported
class CustomModel(LogisticRegressionModel):
    pass


if __name__ == "__main__":

    def _make_data() -> Iterable[np.ndarray]:
        iris = load_iris(as_frame=True)
        x = pd.DataFrame(iris["data"])
        y = pd.DataFrame(iris["target"])

        kept_rows = y["target"] < 2
        x = x.loc[kept_rows, :]
        y = y.loc[kept_rows, :]

        return train_test_split(x, y, test_size=0.2)

    train_x, test_x, train_y, test_y = _make_data()

    model = CustomModel(
        model=None,
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
            "accuracy": accuracy_score,
        }
    )

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

    pipeline = Pipeline([Pipe("scale", CustomStandardScaler, inputs=["x"], outputs=["x"])])
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
        additional_files_to_store=[this_file, this_file.parent / "custom_processer.py"],
    )

    metrics = experiment.run()

    print(metrics)

    # Try loading results again

    folder = Path("outputs")

    inferencer = Inferencer.from_folder(folder)

    train_x, test_x, train_y, test_y = _make_data()
    test_data = DataSetSource(
        x=DataFrameSource(test_x),
        y=DataFrameSource(test_y),
    )
    result = inferencer.predict(test_data)
    print(result)
