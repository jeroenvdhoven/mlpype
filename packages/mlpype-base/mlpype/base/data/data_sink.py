"""An interface for writing data."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

Data = TypeVar("Data")


class DataSink(ABC, Generic[Data]):
    """An interface for writing data."""

    @abstractmethod
    def write(self, data: Data) -> None:
        """Writes data to a given source."""
        raise NotImplementedError
