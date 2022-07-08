from abc import ABC, abstractmethod
from typing import Any, Iterable

from pype.sklearn.data.sklearn_data import SklearnData


class SklearnModelBaseType(ABC):
    @abstractmethod
    def fit(self, *x: SklearnData) -> Any:
        pass

    @abstractmethod
    def predict(self, *x: SklearnData) -> Iterable[SklearnData] | SklearnData:
        pass
