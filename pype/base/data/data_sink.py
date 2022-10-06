from abc import ABC, abstractmethod
from typing import Generic, TypeVar

Data = TypeVar("Data")


class DataSink(ABC, Generic[Data]):
    @abstractmethod
    def write(self, data: Data) -> None:
        """Writes data to a given source."""
        raise NotImplementedError
