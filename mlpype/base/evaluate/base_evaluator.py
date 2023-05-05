from abc import ABC, abstractmethod
from typing import Dict, Generic, Optional, TypeVar, Union

from mlpype.base.data import DataSet
from mlpype.base.model import Model
from mlpype.base.pipeline import Pipeline

Data = TypeVar("Data")


class BaseEvaluator(Generic[Data], ABC):
    def __init__(
        self,
    ) -> None:
        """Evaluates a Model on the given data."""

    @abstractmethod
    def evaluate(
        self, model: Model, data: DataSet, pipeline: Optional[Pipeline] = None
    ) -> Dict[str, Union[str, float, int, str, bool]]:
        """Evaluate the given model on the given dataset.

        We assume the model does not need to be transformed anymore if pipeline is None.

        Args:
            model (Model): The Model to evaluate.
            data (DataSet): The Dataset to use to evaluate the model.
            pipeline (Optional[Pipeline]): If not None, this will be used to transform `data` first.

        Returns:
            Dict[str, Union[str, float, int, str, bool]]: A dictionary of metric_name-value pairs. The result
                of the evaluation.
        """

    @abstractmethod
    def __str__(self, indents: int = 0) -> str:
        """Create string representation of this Evaluator.

        Args:
            indents (int, optional): The number of preceding tabs. Defaults to 0.

        Returns:
            str: A string representation of this Evaluator.
        """
