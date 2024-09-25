"""Provides a generic class for sklearn-like Models."""
import typing
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


class SklearnModel(Model[SklearnData], Generic[T]):
    """A generic class for sklearn-like Models."""

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

        You should set a sklearn model as a type hint to this class when defining a new model.
        This allows us to get the parameters from the documentation of that sklearn model.
        For an example, see the implementation of LinearModel, especially the `SklearnModel[LinearRegression]` part.

        Below are some examples for how to do this yourself.
        ```python
        # Works
        class LinearRegressionModel(SklearnModel[LinearRegression]):
            pass

        # An alternative to dynamically generate the model, which is easier to export/import
        # create_sklearn_model_class can be found in this file.
        model_class = create_sklearn_model_class(LinearRegression)

        # Unfortunately, using something like the following will not work due to how Generic types are handled.
        LinearRegressionModel = SklearnModel[LinearRegression]
        ```

        Args:
            inputs (List[str]): A list of names of input Data. This determines which Data is
                used to fit the model.
            outputs (List[str]): A list of names of output Data. This determines the names of
                output variables.
            model (Optional[T]): An object that has fit() and predict() methods. If none,
                we will use the model_args to instantiate a new model. Should be of type SklearnModelBaseType
            seed (int, optional): The RNG seed to ensure reproducability.. Defaults to 1.
            **model_args (Any): Optional keyword arguments passed to the model class to instantiate a new
                model if `model` is None.
        """
        super().__init__(inputs, outputs, seed)
        if model is None:
            model = self._init_model(model_args)
        self.model = model

    def _init_model(self, args: Dict[str, Any]) -> T:
        return self._get_annotated_class()(**args)

    @classmethod
    def _get_annotated_class(cls) -> Type[T]:
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

    @classmethod
    def class_from_sklearn_model_class(
        cls,
        model_class: Type[SklearnModelBaseType],
    ) -> Type["SklearnModel"]:
        """Create a SklearnModel classfrom a SklearnModelBaseType.

        This should support all sklearn classifaction and regression models.

        Args:
            model_class (Type[SklearnModelBaseType]): The class of the sklearn model. For example,
                LinearRegression or LogisticRegression.

        Returns:
            Type[SklearnModel]: The created SklearnModel.
        """

        class SklearnConditionedModel(cls[model_class]):  # type: ignore
            pass

        return SklearnConditionedModel

    @classmethod
    def from_sklearn_model_class(
        cls,
        model_class: Type[SklearnModelBaseType],
        inputs: List[str],
        outputs: List[str],
        seed: int = 1,
        **model_args: Any,
    ) -> "SklearnModel":
        """Create a SklearnModel from a SklearnModelBaseType.

        This should support all sklearn classifaction and regression models.

        Args:
            model_class (Type[SklearnModelBaseType]): The class of the sklearn model. For example,
                LinearRegression or LogisticRegression.
            inputs (List[str]): A list of names of input Data. This determines which Data is
                used to fit the model.
            outputs (List[str]): A list of names of output Data. This determines the names of
                output variables.
            seed (int, optional): The RNG seed to ensure reproducability.. Defaults to 1.
            **model_args (Any): Optional keyword arguments passed to the model class to instantiate a new
                model if `model` is None.

        Returns:
            SklearnModel: The created SklearnModel.
        """
        new_cls = cls.class_from_sklearn_model_class(model_class)
        return new_cls(inputs=inputs, outputs=outputs, model=None, seed=seed, **model_args)
