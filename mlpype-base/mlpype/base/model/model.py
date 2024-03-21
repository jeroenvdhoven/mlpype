import json
from abc import ABC, abstractclassmethod, abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import Generic, List, Tuple, TypeVar, Union

from mlpype.base.constants import Constants
from mlpype.base.data import DataSet
from mlpype.base.serialiser.joblib_serialiser import JoblibSerialiser

Data = TypeVar("Data")


class Model(ABC, Generic[Data]):
    def __init__(self, inputs: List[str], outputs: List[str], seed: int = 1) -> None:
        """An abstraction of a ML model.

        This should provide the basic interface for fitting, inference, and serialisation.

        Args:
            inputs (List[str]): A list of names of input Data. This determines which Data is
                used to fit the model.
            outputs (List[str]): A list of names of output Data. This determines the names of
                output variables.
            seed (int, optional): The RNG seed to ensure reproducability.. Defaults to 1.
        """
        super().__init__()
        self.seed = seed
        self.inputs = inputs
        self.outputs = outputs
        self.set_seed()

    @classmethod
    def get_parameters(cls, parser: ArgumentParser) -> None:
        """Get and add parameters to initialise this class.

        Args:
            parser (ArgumentParser): The ArgumentParser to add arguments to.
        """

    @abstractmethod
    def set_seed(self) -> None:
        """Sets the RNG seed."""
        raise NotImplementedError

    def save(self, folder: Union[str, Path]) -> None:
        """Stores this model to the given folder.

        This function stores the common inputs and outputs list to the given folder,
        and makes sure it's created. It also stores the model class in the given folder.
        It will call _save to allow further models to specify how they are stored.

        Args:
            folder (Union[str, Path]): The folder to store the Model in.
        """
        joblib_serialiser = JoblibSerialiser()

        if isinstance(folder, str):
            folder = Path(folder)

        folder.mkdir(exist_ok=True, parents=True)
        lists_to_save = {
            "inputs": self.inputs,
            "outputs": self.outputs,
        }
        with open(folder / Constants.MODEL_PARAM_FILE, "w") as f:
            json.dump(lists_to_save, f)

        joblib_serialiser.serialise(self.__class__, folder / Constants.MODEL_CLASS_FILE)
        self._save(folder)

    @abstractmethod
    def _save(self, folder: Path) -> None:
        """Stores this model to the given folder.

        Specifically intended to store

        Args:
            folder (Path): The folder to store the Model in.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, folder: Union[str, Path]) -> "Model":
        """Loads a model from file into this Model.

        Args:
            folder (Union[str, Path]): The folder to load the model from.
        """
        if isinstance(folder, str):
            folder = Path(folder)

        with open(folder / Constants.MODEL_PARAM_FILE) as f:
            lists = json.load(f)

        joblib_serialiser = JoblibSerialiser()
        model_class = joblib_serialiser.deserialise(folder / Constants.MODEL_CLASS_FILE)
        return model_class._load(folder, lists["inputs"], lists["outputs"])

    @abstractclassmethod
    def _load(cls, folder: Path, inputs: List[str], outputs: List[str]) -> "Model":
        """Loads a model from file into this Model.

        Args:
            folder (Union[str, Path]): The folder to load the model from.
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
    def _transform(self, *data: Data) -> Union[Tuple[Data], Data]:
        raise NotImplementedError

    def __str__(self) -> str:
        """Create string representation of this Model.

        Returns:
            str: A string representation of this Model.
        """
        return f"{type(self).__name__}: {self.inputs} -> {self.outputs}"
