from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Serialiser(ABC):
    def __init__(self) -> None:
        """A base class for Serialisers, used to log extra files in Experiments."""
        super().__init__()

    @abstractmethod
    def serialise(self, object: Any, file: str | Path) -> None:
        """Serialise the given object to the given file.

        Args:
            object (Any): The object to serialise.
            file (str | Path): The file to serialise to.
        """
        raise NotImplementedError

    @abstractmethod
    def deserialise(self, file: str | Path) -> Any:
        """Deserialise the object in the given file.

        Args:
            file (str | Path): The file to deserialise.

        Returns:
            Any: The python object stored in the file.
        """
        raise NotImplementedError
