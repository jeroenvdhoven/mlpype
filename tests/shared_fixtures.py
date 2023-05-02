from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

from pytest import fixture

from pype.base.experiment.experiment import Experiment
from pype.base.logger.local_logger import LocalLogger
from pype.base.serialiser.joblib_serialiser import JoblibSerialiser
from tests.utils import (
    DummyModel,
    get_dummy_data,
    get_dummy_evaluator,
    get_dummy_pipeline,
    get_dummy_type_checkers,
)


@fixture(scope="module")
def dummy_experiment() -> Iterable[Experiment]:
    train = get_dummy_data(20, 1, 0)
    test = get_dummy_data(5, 2, -1)

    input_checker, output_checker = get_dummy_type_checkers()

    with TemporaryDirectory() as f:
        output_folder = Path(f) / "tmp_folder"
        yield Experiment(
            data_sources={"train": train, "test": test},
            model=DummyModel(inputs=["x"], outputs=["y"]),
            pipeline=get_dummy_pipeline(),
            logger=LocalLogger(),
            evaluator=get_dummy_evaluator(),
            serialiser=JoblibSerialiser(),
            output_folder=output_folder,
            input_type_checker=input_checker,
            output_type_checker=output_checker,
            parameters={"param_1": 2},
        )
