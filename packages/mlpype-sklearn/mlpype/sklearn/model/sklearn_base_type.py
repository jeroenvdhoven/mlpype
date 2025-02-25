"""Provides a generic class for sklearn-like Models."""
from typing import Any, Iterable, Protocol, Union

from mlpype.sklearn.data.sklearn_data import SklearnData


class SklearnModelBaseType(Protocol):
    """Base class for sklearn-like models."""

    def fit(self, *x: SklearnData, **kwargs: Any) -> Any:
        """Fit a model to the given data. Kwargs are ignored."""

    def predict(self, *x: SklearnData, **kwargs: Any) -> Union[Iterable[SklearnData], SklearnData]:
        """Predict for given data using a trained model. Kwargs are ignored."""
