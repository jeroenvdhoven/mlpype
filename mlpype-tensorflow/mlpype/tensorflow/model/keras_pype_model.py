import typing
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Type, TypeVar, Union

from keras import Model as KerasBaseModel
from keras.losses import Loss
from keras.metrics import Metric
from keras.models import load_model as keras_load_model
from keras.optimizers import Optimizer
from tensorflow import Tensor  # type: ignore
from tensorflow.data import Dataset  # type: ignore
from tensorflow.random import set_seed  # type: ignore

from mlpype.base.experiment.argument_parsing import add_args_to_parser_for_class
from mlpype.base.model import Model
from mlpype.base.serialiser.joblib_serialiser import JoblibSerialiser

T = TypeVar("T", bound=KerasBaseModel)


class KerasPypeModel(Model[Tensor], ABC, Generic[T]):
    KERAS_MODEL_FILE = "keras_model"
    LOSS_FILE = "loss_func.pkl"

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        loss: Union[Callable, Loss],
        optimizer_class: Type[Optimizer],
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        metrics: Optional[List[Union[Metric, Dict[str, Metric]]]] = None,
        model: Optional[T] = None,
        seed: int = 1,
        **model_args: Any,
    ) -> None:
        """A Model to integrate Keras models with the mlpype framework.

        Args:
            inputs (List[str]): Input names from the DataSet to use.
            outputs (List[str]): Output names from the DataSet to use.
            loss (Union[Callable, Loss]): The loss for your Keras Model.
            optimizer_class (Type[Optimizer]): The optimization class to use.
            epochs (int, optional): The number of epochs to use for training. Defaults to 10.
            batch_size (int, optional): The batch size to use for training. Defaults to 32.
            learning_rate (float, optional): The learning rate to use for training. Defaults to 0.001.
            metrics (Optional[List[Union[Metric, Dict[str, Metric]]]]): Additional metrics to
                use while training th emodel. Defaults to no metrics.
            model (Optional[T]): The Keras Model to train. By Default we'll use excess arguments
                to initialise the default model.
            seed (int, optional): The seed to initialise tensorflow with. Defaults to 1.
        """
        super().__init__(inputs, outputs, seed)
        if metrics is None:
            metrics = []

        if model_args is None:
            model_args = {}

        if model is None:
            model = self._init_model(model_args)
        self.model = model
        self.optimizer = optimizer_class(learning_rate=learning_rate)
        self.loss = loss
        self.metrics = metrics

        self.epochs = epochs
        self.batch_size = batch_size

    @abstractmethod
    def _init_model(self, args: Dict[str, Any]) -> T:
        raise NotImplementedError

    @classmethod
    def _get_annotated_class(cls) -> Type[KerasBaseModel]:
        return typing.get_args(cls.__orig_bases__[0])[0]

    def set_seed(self) -> None:
        """Sets the RNG seed."""
        set_seed(self.seed)

    def _fit(self, *data: Tensor) -> None:
        dataset = Dataset.from_tensor_slices(data).batch(self.batch_size)

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        self.model.fit(dataset, epochs=self.epochs)

    def _transform(self, *data: Tensor) -> Union[Iterable[Tensor], Tensor]:
        return self.model(*data)

    def _save(self, folder: Path) -> None:
        serialiser = JoblibSerialiser()
        serialiser.serialise(self.loss, folder / self.LOSS_FILE)
        self.model.save(folder / self.KERAS_MODEL_FILE)

    @classmethod
    def _load(cls: Type["KerasPypeModel"], folder: Path, inputs: List[str], outputs: List[str]) -> "KerasPypeModel":
        model: KerasBaseModel = keras_load_model(folder / cls.KERAS_MODEL_FILE)

        serialiser = JoblibSerialiser()
        loss = serialiser.deserialise(folder / cls.LOSS_FILE)
        optimizer = model.optimizer
        return cls(inputs=inputs, outputs=outputs, model=model, seed=1, loss=loss, optimizer_class=optimizer.__class__)

    @classmethod
    def get_parameters(cls: Type["KerasPypeModel"], parser: ArgumentParser) -> None:
        """Get and add parameters to initialise this class.

        KerasPypeModel's will work by requiring 2 ways to instantiate a Model:
            - through `model`, which is a keras model.
            - through parameters, which will instantiate the model internally.

        Arguments are taken from both the Keras Model class and the mlpype Model class.

        Args:
            parser (ArgumentParser): The ArgumentParser to add arguments to.
        """
        super().get_parameters(parser)
        BaseModel = cls._get_annotated_class()

        add_args_to_parser_for_class(parser, BaseModel, "model", [KerasBaseModel], excluded_args=[])

        add_args_to_parser_for_class(
            parser,
            cls,
            "model",
            [Model],
            excluded_args=["seed", "inputs", "outputs", "model", "loss", "optimizer_class"],
        )
