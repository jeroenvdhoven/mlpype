from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar

Data = TypeVar("Data")


class Operator(ABC, Generic[Data]):
    def __init__(self) -> None:
        """An Operator that can been fitted to Data and transform new data.

        This is not meant to be extended, but should work well enough to help
        people identify what their Pipes need as input.
        """
        super().__init__()

    @abstractmethod
    def fit(self, *data: Data) -> "Operator":
        """Fit the Operator to the given Data.

        Returns:
            Operator: this object.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, *data: Data) -> Tuple[Data, ...]:
        """Transforms the data using this Operator.

        You do not need to return the same amount of Data as you
        used for input.

        Returns:
            Tuple[Data]: A Tuple of Data elements, the transformed
                Data.
        """
        raise NotImplementedError

    def inverse_transform(self, *data: Data) -> Tuple[Data, ...]:
        """Optionally, you can include an option to inverse transform the data.

        Returns:
            Tuple[Data]: An Tuple of Data elements, the inverse transformed
                Data.
        """
        return data
