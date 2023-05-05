from abc import ABC, abstractmethod
from typing import Any, Iterable, Union

from mlpype.sklearn.data.sklearn_data import SklearnData


class SklearnModelBaseType(ABC):
    @abstractmethod
    def fit(self, *x: SklearnData) -> Any:
        """Fit a model to the given data."""

    @abstractmethod
    def predict(self, *x: SklearnData) -> Union[Iterable[SklearnData], SklearnData]:
        """Predict for given data using a trained model."""
