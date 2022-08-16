from abc import ABC
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Type

import numpy as np

from pype.base.experiment.argument_parsing import add_args_to_parser_for_class
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
        """A generic class for sklearn-like Models.

        Args:
            inputs (List[str]): A list of names of input Data. This determines which Data is
                used to fit the model.
            outputs (List[str]): A list of names of output Data. This determines the names of
                output variables.
            model (SklearnModelBaseType): An object that has fit() and predict() methods.
            seed (int, optional): The RNG seed to ensure reproducability.. Defaults to 1.
        """
        super().__init__(inputs, outputs, seed)
        self.model = model

    def set_seed(self) -> None:
        """Sets the RNG seed."""
        np.random.seed(self.seed)

    def _fit(self, *data: SklearnData) -> None:
        self.model.fit(*data)

    def _transform(self, *data: SklearnData) -> Iterable[SklearnData] | SklearnData:
        return self.model.predict(*data)

    def _save(self, folder: Path) -> None:
        serialiser = JoblibSerialiser()
        serialiser.serialise(self.model, folder / self.SKLEARN_MODEL_FILE)

    @classmethod
    def _load(cls: Type["SklearnModel"], folder: Path, inputs: list[str], outputs: list[str]) -> "SklearnModel":
        serialiser = JoblibSerialiser()
        model = serialiser.deserialise(folder / cls.SKLEARN_MODEL_FILE)
        return cls(inputs=inputs, outputs=outputs, model=model, seed=1)

    @classmethod
    def get_parameters(cls: Type["SklearnModel"], parser: ArgumentParser) -> None:
        """Get and add parameters to initialise this class.

        SklearnModel's will work by requiring 2 ways to instantiate a Model:
            - through `model`, which is a sklearn model.
            - through parameters, which will instantiate the model internally.

        Args:
            parser (ArgumentParser): The ArgumentParser to add arguments to.
        """
        super().get_parameters(parser)
        add_args_to_parser_for_class(
            parser, cls, "model", [SklearnModel], excluded_args=["seed", "inputs", "outputs", "model"]
        )
