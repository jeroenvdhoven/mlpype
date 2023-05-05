from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union


class Serialiser(ABC):
    def __init__(self) -> None:
        """A base class for Serialisers, used to log extra files in Experiments.

        Please make sure any Serialiser you create can be deserialised using a JoblibSerialiser.
        These are used in the Experiment/Inferencer to save/load your Serialisers.
        """
        super().__init__()

    @abstractmethod
    def serialise(self, object: Any, file: Union[str, Path]) -> None:
        """Serialise the given object to the given file.

        Args:
            object (Any): The object to serialise.
            file (Union[str, Path]): The file to serialise to.
        """
        raise NotImplementedError

    @abstractmethod
    def deserialise(self, file: Union[str, Path]) -> Any:
        """Deserialise the object in the given file.

        Args:
            file (Union[str, Path]): The file to deserialise.

        Returns:
            Any: The python object stored in the file.
        """
        raise NotImplementedError
