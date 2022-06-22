from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Iterable, List

from pype.base.data import DataSet
from pype.base.data.data import Data


class Model(ABC, Generic[Data]):
    def __init__(self, seed: int, inputs: List[str], outputs: List[str]) -> None:
        """An abstraction of a ML model.

        This should provide the basic interface for fitting, inference, and serialisation.

        Args:
            seed (int): The RNG seed to ensure reproducability.
            inputs (List[str]): A list of names of input Data. This determines which Data is
                used to fit the model.
            outputs (List[str]): A list of names of output Data. This determines the names of
                output variables.
        """
        super().__init__()
        self.seed = seed
        self.inputs = inputs
        self.outputs = outputs
        self.set_seed()

    @abstractmethod
    def set_seed(self) -> None:
        """Sets the RNG seed."""
        raise NotImplementedError

    @abstractmethod
    def save(self, file: str | Path) -> None:
        """Stores this model to the given file.

        Args:
            file (str | Path): The file to store the Model in.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, file: str | Path) -> "Model":
        """Loads a model from file into this Model.

        Args:
            file (str | Path): The file to load the model from.
        """
        raise NotImplementedError

    def fit(self, data: DataSet) -> "Model":
        """Fits the Model to the given DataSet.

        The DataSet should contain all inputs and outputs.

        Args:
            data (DataSet): The DataSet to fit this Model on.
        """
        self._fit(*data.get_all(self.inputs), *data.get_all(self.outputs))
        return self

    @abstractmethod
    def _fit(self, *data: Data) -> None:
        raise NotImplementedError

    def transform(self, data: DataSet) -> DataSet:
        """Applies the Model to the given DataSet.

        The DataSet should contain all inputs.

        Args:
            data (DataSet): The DataSet to transform using this Model.

        Returns:
            DataSet: The outputs as a Dataset.
        """
        result = self._transform(*data.get_all(self.inputs))
        if len(self.outputs) == 1:
            result = [result]  # type: ignore

        # type check handled by above check.
        return DataSet.from_dict({name: data for name, data in zip(self.outputs, result)})  # type: ignore

    @abstractmethod
    def _transform(self, *data: Data) -> Iterable[Data] | Data:
        raise NotImplementedError
