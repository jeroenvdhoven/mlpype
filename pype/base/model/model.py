from abc import ABC, abstractmethod
from typing import Generic, List, Tuple

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
    def save(self, file: str) -> None:
        """Stores this model to the given file.

        Args:
            file (str): The file to store the Model in.
        """
        raise NotImplementedError

    @abstractmethod
    @classmethod
    def load(cls, file: str) -> "Model":
        """Loads a model from file into this Model.

        Args:
            file (str): The file to load the model from.
        """
        raise NotImplementedError

    def fit(self, data: DataSet) -> None:
        """Fits the Model to the given DataSet.

        The DataSet should contain all inputs and outputs.

        Args:
            data (DataSet): The DataSet to fit this Model on.
        """
        self._fit(*data.get_all(self.inputs), *data.get_all(self.outputs))

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
        return DataSet({name: data for name, data in zip(self.outputs, result)})

    @abstractmethod
    def _transform(self, *data: Data) -> Tuple[Data, ...]:
        raise NotImplementedError
