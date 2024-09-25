"""Please run this file using `python -m examples.custom_classes.custom_model_example`.

We do not guarantee results if you use `python examples/custom_classes/custom_model_example.py`
"""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
from mlpype.sklearn.model import LogisticRegressionModel

from .dummy_model.custom_processer import CustomStandardScaler, tcc


# classes / functions defined in the core file can be imported
class CustomModel(LogisticRegressionModel):
    pass


output_folder = Path(__file__).parent / "__outputs"


def _make_data() -> Iterable[np.ndarray]:
    iris = load_iris(as_frame=True)
    x = pd.DataFrame(iris["data"])
    y = pd.DataFrame(iris["target"])

    kept_rows = y["target"] < 2
    x = x.loc[kept_rows, :]
    y = y.loc[kept_rows, :]

    return train_test_split(x, y, test_size=0.2)


if __name__ == "__main__":
    train_x, test_x, train_y, test_y = _make_data()

    model = CustomModel(
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
        output_folder=output_folder,
        # Need to add this file to output to make sure we can import CustomModel and CustomStandardScaler/tcc
        additional_files_to_store=[this_file, this_file.parent / "dummy_model"],
    )

    metrics = experiment.run()

    print(metrics)

    # Try loading results again

    folder = Path("outputs")

    inferencer = Inferencer.from_folder(folder)
    test_data = DataCatalog(
        x=DataFrameSource(test_x),
    )
    result = inferencer.predict(test_data)
    print(result)
