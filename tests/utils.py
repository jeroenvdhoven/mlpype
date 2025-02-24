from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type, Union

from mlpype.base.data.data_catalog import DataCatalog
from mlpype.base.data.data_sink import DataSink
from mlpype.base.data.data_source import DataSource
from mlpype.base.evaluate.evaluator import Evaluator
from mlpype.base.model import Model
from mlpype.base.pipeline.operator import Operator
from mlpype.base.pipeline.pipe import Pipe
from mlpype.base.pipeline.pipeline import Pipeline
from mlpype.base.pipeline.type_checker import DataModel, DataSetModel, TypeChecker, TypeCheckerPipe
from tests.utils_training_support import reverse


class AnyArg:
    def __eq__(self, __o: object) -> bool:
        return True


@contextmanager
def pytest_assert(error_class, message: Optional[str] = None, exact: bool = True):
    try:
        yield
        raise ValueError("No error was raised!")
    except error_class as e:
        if message is not None:
            error_message: str = e.args[0]
            if exact:
                assert (
                    error_message == message
                ), f"Error messages did not match: Got {error_message}, expected {message}"
            else:
                assert (
                    message in error_message
                ), f"Error messages did not match: Got {error_message}, expected {message}"


class DummyModel(Model[List[Union[int, float]]]):
    mean_file = "mean.txt"

    def __init__(self, inputs: List[str], outputs: List[str], seed: int = 1, a: int = 3, b: float = 5) -> None:
        super().__init__(inputs, outputs, seed)
        self.a = a
        self.b = b

    def set_seed(self) -> None:
        pass

    def _fit(self, x: List[Union[int, float]], y: List[Union[int, float]]) -> None:
        self.prediction = sum(y) / len(y) + self.a

    def _transform(self, x: List[Union[int, float]]) -> List[Union[int, float]]:
        return [self.prediction for _ in x]

    def _save(self, folder: Path) -> None:
        with open(folder / self.mean_file, "w") as f:
            f.write(str(self.prediction))

    @classmethod
    def _load(cls, folder: Path, inputs: List[str], outputs: List[str]) -> "DummyModel":
        result = cls(inputs=inputs, outputs=outputs)
        with open(folder / cls.mean_file, "r") as f:
            result.prediction = float(f.read())
        return result


class DummyDataSource(DataSource[List[float]]):
    def __init__(self, l) -> None:
        super().__init__()
        self.l = l

    def read(self) -> List[float]:
        return self.l

    def __eq__(self, __o: object) -> bool:
        return self.l == __o.l


def get_dummy_data(n: int, x_offset: int, y_offset: int) -> DataCatalog:
    return DataCatalog(
        x=DummyDataSource([i + x_offset for i in range(n)]),
        y=DummyDataSource([i + y_offset for i in range(n)]),
    )


class DummyDataSink(DataSink[List[float]]):
    def __init__(self) -> None:
        self.data = None

    def write(self, data: List[float]) -> None:
        self.data = data


class DummyOperator(Operator[List[float]]):
    def __init__(self, c: int = 0) -> None:
        super().__init__()
        self.c = c

    def fit(self, x: List[float]) -> "Operator":
        return self

    def transform(self, x: List[float]) -> List[float]:
        return reverse([i - 1 for i in x])


def get_dummy_pipeline() -> Pipeline:
    return Pipeline([Pipe(name="minus 1", operator=DummyOperator, inputs=["x"], outputs=["y"])])


def get_dummy_evaluator() -> Evaluator:
    return Evaluator({"diff": lambda x, y: (sum([i - j for i, j in zip(x, y)]) / len(y))})


class DummyDataModel(DataModel):
    data: List[float]

    def convert(self) -> List[float]:
        return self.data

    @classmethod
    def to_model(cls, data: List[float]) -> "DataModel":
        return cls(data=data)


class DummyDataSet(DataSetModel):
    x: DummyDataModel
    y: DummyDataModel


class DummyTypeChecker(TypeChecker):
    def fit(self, data: List[float]) -> "Operator":
        return super().fit(data)

    def transform(self, data: List[float]) -> List[float]:
        assert isinstance(data, list), "Provide a list!"
        assert isinstance(data[0], float) or isinstance(data[0], int), "Provide a list with ints/floats!"
        return data

    def get_pydantic_type(self) -> Type[DataModel]:
        return DummyDataModel

    @classmethod
    def supports_object(cls, obj: Any) -> bool:
        return isinstance(obj, list)


def get_dummy_type_checkers() -> Tuple[TypeCheckerPipe, TypeCheckerPipe]:
    return TypeCheckerPipe("input", ["x"], type_checker_classes=[DummyTypeChecker]), TypeCheckerPipe(
        "output", ["y"], type_checker_classes=[DummyTypeChecker]
    )
