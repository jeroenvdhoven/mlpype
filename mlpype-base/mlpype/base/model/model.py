"""Provides the core Model class, which is the base class for all mlpype-compliant models."""
import json
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import Generic, List, Tuple, TypeVar, Union

from dill import dump, load

from mlpype.base.constants import Constants
from mlpype.base.data import DataSet

Data = TypeVar("Data")


class Model(ABC, Generic[Data]):
    """An abstraction of a ML model.

    This class is the core for any Models to integrate with the mlpype framework.
    This is the main abstraction layer between most other packages like sklearn and
    mlpype. It allows you to train models and make predictions using the same interface,
    so that you can easily switch between packages and thus models.

    Extend it by implementing the abstract methods:

    - `_fit`: Fit the model to the given data.
    - `_transform`: Transform the given data using the model to make predictions.
    - `_save`: Save the model to the given folder.
    - `_load`: Load the model from the given folder.
    - `set_seed`: Set the RNG seed.

    """

    def __init__(self, inputs: List[str], outputs: List[str], seed: int = 1) -> None:
        """An abstraction of a ML model.

        This should provide the basic interface for fitting, inference, and serialisation.

        Args:
            inputs (List[str]): A list of names of input Data. This determines which Data is
                used to fit the model.
            outputs (List[str]): A list of names of output Data. This determines the names of
                output variables.
            seed (int, optional): The RNG seed to ensure reproducability. Defaults to 1.
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
        """Sets the RNG seed.

        Use this to ensure reproducability.
        """
        raise NotImplementedError

    def save(self, folder: Union[str, Path]) -> None:
        """Stores this model to the given folder.

        This function stores the common inputs and outputs list to the given folder,
        and makes sure it's created. It also stores the model class in the given folder.
        It will call _save to allow further models to specify how they are stored.

        `_save` will be called with the same folder argument as this function.

        Args:
            folder (Union[str, Path]): The folder to store the Model in.
        """
        if isinstance(folder, str):
            folder = Path(folder)

        folder.mkdir(exist_ok=True, parents=True)
        lists_to_save = {
            "inputs": self.inputs,
            "outputs": self.outputs,
        }
        with open(folder / Constants.MODEL_PARAM_FILE, "w") as f:
            json.dump(lists_to_save, f)

        with open(folder / Constants.MODEL_CLASS_FILE, "wb") as f:
            dump(self.__class__, f)
        self._save(folder)

    @abstractmethod
    def _save(self, folder: Path) -> None:
        """Stores this model to the given folder.

        Specifically intended to store the real model artifact, like the sklearn
        model object or keras model.

        Args:
            folder (Path): The folder to store the Model in.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, folder: Union[str, Path]) -> "Model":
        """Loads a model from a folder into this Model.

        This function first loads the common inputs and outputs list from the given folder,
        then loads the model class in the given folder. It will call _load on this class to allow
        models to specify how they are loaded.

        This is inteded to be called like this:

        ```python
        model = Model.load(folder)
        ```

        Args:
            folder (Union[str, Path]): The folder to load the model from.

        Returns:
            Model: The loaded model. It's type is determined by the model class, which should
                be specified in the folder.
        """
        if isinstance(folder, str):
            folder = Path(folder)

        with open(folder / Constants.MODEL_PARAM_FILE) as f:
            lists = json.load(f)

        with open(folder / Constants.MODEL_CLASS_FILE, "rb") as f:
            model_class = load(f)
        return model_class._load(folder, lists["inputs"], lists["outputs"])

    @classmethod
    @abstractmethod
    def _load(cls, folder: Path, inputs: List[str], outputs: List[str]) -> "Model":
        """Loads a model from file into this Model.

        Generally this focusses only on loading the actual model artifact, like the sklearn
        model object or keras model.

        Args:
            folder (Path): The folder to load the model from.
            inputs (List[str]): A list of names of input Data. This determines which Data is
                used to fit the model.
            outputs (List[str]): A list of names of output Data. This determines the names of
                output variables.

        Returns:
            Model: self
        """
        raise NotImplementedError

    def fit(self, data: DataSet) -> "Model":
        """Fits the Model to the given DataSet.

        The DataSet should contain all inputs and outputs. This calls the `_fit` function
        to actually fit the model, which handles the actual implementation. On the other hand,
        this function handles grabbing the correct inputs and outputs from the DataSet. This allows
        the implementations of `_fit` to just use raw data, like 2 numpy arrays for X and y.

        Args:
            data (DataSet): The DataSet to fit this Model on.

        Returns:
            Model: This model.
        """
        self._fit(*data.get_all(self.inputs), *data.get_all(self.outputs))
        return self

    @abstractmethod
    def _fit(self, *data: Data) -> None:
        """Fits the Model to the given data.

        This function can directly use the raw data to fit the model. For instance, in the
        case of a sklearn model, it can directly use the raw data to fit the model:

        ```python
        def _fit(self, X, y):
            # Assume self.model is a sklearn model
            return self.model.fit(X, y)
        ```

        This is compatible with the MLPype API.

        Args:
            *data (Data): The data to fit using this Model.
        """
        raise NotImplementedError

    def transform(self, data: DataSet) -> DataSet:
        """Applies the Model to the given DataSet.

        The DataSet should contain all inputs. This calls the `_transform` function
        to actually apply the model, which handles the actual implementation. On the other hand,
        this function handles grabbing the correct inputs from the DataSet. This allows
        the implementations of `_transform` to just use raw data. It also makes sure
        that the output is a DataSet by combining it with the `outputs`.

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
        """Applies the Model to the given data.

        This function can use raw data, like 2 numpy arrays for X and y. In the case of
        a sklearn model, you could define `_transform` like:
        ```python
        def _transform(self, X, y):
            # Assume self.model is a sklearn model
            return self.model.fit(X, y)
        ```

        This will be compatible with the MLPype API.

        Args:
            *data (Data): The DataSet to transform using this Model.

        Returns:
            Union[Tuple[Data], Data]: Either a tuple of Data or a single Data object.
                For instance, this can directly return the predictions of a sklearn model.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """Create string representation of this Model.

        Returns:
            str: A string representation of this Model.
        """
        return f"{type(self).__name__}: {self.inputs} -> {self.outputs}"
