from typing import Callable, Dict, Generic, Tuple

from pype.base.data import DataSet
from pype.base.data.data import Data
from pype.base.model import Model
from pype.base.pipeline import Pipeline


class Evaluator(Generic[Data]):
    def __init__(
        self,
        functions: Dict[str, Callable[[Tuple[Data, ...]], float | int | str | bool]],
    ) -> None:
        """Evaluates a Model on the given Functions.

        Args:
            functions (Dict[str, Callable[[Tuple[Data, ...]], float | int | str | bool]]):
                A dict of metric names and metric functions. We expect each to return
                a float, int, string, or boolean value. We provide the arguments as follows:
                    - first, all labeled data, e.g. y_true.
                    - second, all predicted data, e.g. y_pred.
        """
        self.functions = functions

    def evaluate(
        self, model: Model, data: DataSet, pipeline: Pipeline | None = None
    ) -> Dict[str, float | int | str | bool]:
        """Evaluate the given model on the given dataset.

        We assume the model does not need to be transformed anymore if pipeline is None.

        Args:
            model (Model): The Model to evaluate.
            data (DataSet): The Dataset to use to evaluate the model.
            pipeline (Pipeline, optional): If not None, this will be used to transform `data` first.

        Returns:
            Dict[str, float | int | str | bool]: A dictionary of metric_name-value pairs. The result
                of the evaluation.
        """
        if pipeline is not None:
            data = pipeline.transform(data)

        predictions = model.transform(data)
        output_data = data.get_all(model.outputs)

        return {name: func(*output_data, *predictions) for name, func in self.functions.items()}
