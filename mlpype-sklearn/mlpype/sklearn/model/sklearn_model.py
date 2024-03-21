import typing
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Generic, Iterable, List, Optional, Type, TypeVar, Union

import numpy as np

from mlpype.base.experiment.argument_parsing import add_args_to_parser_for_class
from mlpype.base.model import Model
from mlpype.base.serialiser.joblib_serialiser import JoblibSerialiser
from mlpype.sklearn.data.sklearn_data import SklearnData
from mlpype.sklearn.model.sklearn_base_type import SklearnModelBaseType

T = TypeVar("T", bound=SklearnModelBaseType)


class SklearnModel(Model[SklearnData], ABC, Generic[T]):
    SKLEARN_MODEL_FILE = "model.pkl"

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        model: Optional[T] = None,
        seed: int = 1,
        **model_args: Any,
    ) -> None:
        """A generic class for sklearn-like Models.

        You can set a sklearn model as a type hint to this class when defining a new model.
        This allows us to get the parameters from the documentation of that sklearn model.
        For an example, see the implementation of LinearModel, especially the `SklearnModel[LinearRegression]` part.

        class LinearRegressionModel(SklearnModel[LinearRegression]):
            def _init_model(self, **args) -> LinearRegression:
                return LinearRegression(**args)


        Args:
            inputs (List[str]): A list of names of input Data. This determines which Data is
                used to fit the model.
            outputs (List[str]): A list of names of output Data. This determines the names of
                output variables.
            model (Optional[SklearnModelBaseType]): An object that has fit() and predict() methods. If none,
                we will use the model_args to instantiate a new model.
            seed (int, optional): The RNG seed to ensure reproducability.. Defaults to 1.
            **model_args (Any): Optional keyword arguments passed to the model class to instantiate a new
                model if `model` is None.
        """
        super().__init__(inputs, outputs, seed)
        if model is None:
            model = self._init_model(model_args)
        self.model = model

    @abstractmethod
    def _init_model(self, args: Dict[str, Any]) -> T:
        raise NotImplementedError

    @classmethod
    def _get_annotated_class(cls) -> Type[SklearnModelBaseType]:
        return typing.get_args(cls.__orig_bases__[0])[0]

    def set_seed(self) -> None:
        """Sets the RNG seed."""
        np.random.seed(self.seed)

    def _fit(self, *data: SklearnData) -> None:
        self.model.fit(*data)

    def _transform(self, *data: SklearnData) -> Union[Iterable[SklearnData], SklearnData]:
        return self.model.predict(*data)

    def _save(self, folder: Path) -> None:
        serialiser = JoblibSerialiser()
        serialiser.serialise(self.model, folder / self.SKLEARN_MODEL_FILE)

    @classmethod
    def _load(cls: Type["SklearnModel"], folder: Path, inputs: List[str], outputs: List[str]) -> "SklearnModel":
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
        BaseModel = cls._get_annotated_class()

        add_args_to_parser_for_class(
            parser, BaseModel, "model", [], excluded_args=["seed", "inputs", "outputs", "model"]
        )
