from abc import ABC, abstractclassmethod
from argparse import ArgumentParser
from functools import wraps
from pathlib import Path
from typing import Iterable, Type

import numpy as np

from pype.base.experiment.parsing import add_args_to_parser_for_function
from pype.base.model import Model
from pype.base.serialiser.joblib_serialiser import JoblibSerialiser
from pype.sklearn.data.sklearn_data import SklearnData
from pype.sklearn.model.sklearn_base_type import SklearnModelBaseType


class SklearnModel(Model[SklearnData], ABC):
    SKLEARN_MODEL_FILE = "model.pkl"

    def __init__(
        self,
        inputs: list[str],
        outputs: list[str],
        model: SklearnModelBaseType,
        seed: int = 1,
    ) -> None:
        super().__init__(inputs, outputs, seed)
        self.model = model

    def set_seed(self) -> None:
        np.random.seed(self.seed)

    def _fit(self, *data: SklearnData) -> None:
        self.model.fit(*data)

    def _transform(self, *data: SklearnData) -> Iterable[SklearnData] | SklearnData:
        return self.model.predict(*data)

    def _save(self, folder: Path) -> None:
        serialiser = JoblibSerialiser()
        serialiser.serialise(self.model, folder / self.SKLEARN_MODEL_FILE)

    @classmethod
    def _load(cls: Type["SklearnModel"], folder: Path, inputs: list[str], outputs: list[str]) -> "Model":
        serialiser = JoblibSerialiser()
        model = serialiser.deserialise(folder / cls.SKLEARN_MODEL_FILE)
        return cls(inputs=inputs, outputs=outputs, model=model, seed=1)

    @abstractclassmethod
    def from_parameters(**kwargs) -> "SklearnModel":
        raise NotImplementedError

    @classmethod
    def get_parameters(cls: Type["SklearnModel"], parser: ArgumentParser) -> None:
        super().get_parameters(parser)
        add_args_to_parser_for_function(parser, cls.from_parameters, "model", excluded=["seed", "inputs", "outputs"])
