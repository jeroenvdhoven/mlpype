from contextlib import contextmanager
from pathlib import Path

from pype.base.data.data_source import DataSource
from pype.base.data.dataset_source import DataSetSource
from pype.base.evaluate.evaluator import Evaluator
from pype.base.experiment.experiment import Experiment
from pype.base.logger.local_logger import LocalLogger
from pype.base.model import Model
from pype.base.pipeline.operator import Operator
from pype.base.pipeline.pipe import Pipe
from pype.base.pipeline.pipeline import Pipeline
from pype.base.pipeline.type_checker import DataModel, TypeChecker, TypeCheckerPipe
from pype.base.serialiser.joblib_serialiser import JoblibSerialiser
from tests.test_utils_training_support import reverse


@contextmanager
def pytest_assert(error_class, message: str | None = None, exact: bool = True):
    try:
        yield
        raise ValueError("No error was raised!")
    except error_class as e:
        if message is not None:
            error_message: str = e.args[0]
            if exact:
                assert error_message == message
            else:
                assert message in error_message


class DummyModel(Model[list[int | float]]):
    mean_file = "mean.txt"

    def set_seed(self) -> None:
        pass

    def _fit(self, x: list[int | float], y: list[int | float]) -> None:
        self.prediction = sum(y) / len(y)

    def _transform(self, x: list[int | float]) -> list[int | float]:
        return [self.prediction for _ in x]

    def _save(self, folder: Path) -> None:
        with open(folder / self.mean_file, "w") as f:
            f.write(str(self.prediction))

    @classmethod
    def _load(cls, folder: Path, inputs: list[str], outputs: list[str]) -> "DummyModel":
        result = cls(inputs=inputs, outputs=outputs)
        with open(folder / cls.mean_file, "r") as f:
            result.prediction = float(f.read())
        return result


class DummyDataSource(DataSource[list[float]]):
    def __init__(self, l) -> None:
        super().__init__()
        self.l = l

    def read(self) -> list[float]:
        return self.l


def get_dummy_data(n: int, x_offset: int, y_offset: int) -> DataSetSource:
    return DataSetSource(
        x=DummyDataSource([i + x_offset for i in range(n)]),
        y=DummyDataSource([i + y_offset for i in range(n)]),
    )


class DummyOperator(Operator[list[float]]):
    def fit(self, x: list[float]) -> "Operator":
        return self

    def transform(self, x: list[float]) -> list[float]:
        return reverse([i - 1 for i in x])


def get_dummy_pipeline() -> Pipeline:
    return Pipeline([Pipe(name="minus 1", operator=DummyOperator, inputs=["x"], outputs=["y"])])


def get_dummy_evaluator() -> Evaluator:
    return Evaluator({"diff": lambda x, y: (sum([i - j for i, j in zip(x, y)]) / len(y))})


class DummyDataModel(DataModel):
    data: list[float]

    def convert(self) -> list[float]:
        return self.data

    @classmethod
    def to_model(cls, data: list[float]) -> "DataModel":
        return cls(data=data)


class DummyTypeChecker(TypeChecker):
    def fit(self, data: list[float]) -> "Operator":
        return super().fit(data)

    def transform(self, data: list[float]) -> list[float]:
        assert isinstance(data, list), "Provide a list!"
        assert isinstance(data[0], float) or isinstance(data[0], int), "Provide a list with ints/floats!"
        return data

    def get_pydantic_type(self) -> type[DataModel]:
        return DummyDataModel


def get_dummy_type_checkers() -> tuple[TypeCheckerPipe, TypeCheckerPipe]:
    return TypeCheckerPipe("input", ["x"], type_checker_classes=[(list, DummyTypeChecker)]), TypeCheckerPipe(
        "output", ["y"], type_checker_classes=[(list, DummyTypeChecker)]
    )


def get_dummy_experiment() -> Experiment:
    train = get_dummy_data(20, 1, 0)
    test = get_dummy_data(5, 2, -1)

    output_folder = Path(__file__).parent / "tmp_folder"

    input_checker, output_checker = get_dummy_type_checkers()

    return Experiment(
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
