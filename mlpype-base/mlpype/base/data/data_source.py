from abc import ABC, abstractmethod
from typing import Generic, TypeVar

Data = TypeVar("Data")


class DataSource(ABC, Generic[Data]):
    @abstractmethod
    def read(self) -> Data:
        """Read data from a given source.

        This method should be build ideally such that it provides consistent
        datasets every time it is called.

        Returns:
            Data: Data to be returned once read.
        """
        raise NotImplementedError
