"""Evaluates a Model on the given Functions."""
from typing import Callable, Dict, Optional, TypeVar, Union

from mlpype.base.data import DataSet
from mlpype.base.evaluate.base_evaluator import BaseEvaluator
from mlpype.base.model import Model
from mlpype.base.pipeline import Pipeline

Data = TypeVar("Data")


class Evaluator(BaseEvaluator[Data]):
    """Evaluates a Model on the given Functions.

    This is the default Evaluator, and should fit most models and packages.
    It assumes you provide functions that take in (y_true, y_pred) and return a float, int, string, or boolean value.

    Most sklearn evaluation metrics and those from similar packages should be compliant with this format, and otherwise
    it's not too difficult to adept them to this format. Spark for instance does things differently and has its
    own implementation. For example, in sklearn you can do:

    ```python
    from mlpype.base.evaluate import Evaluator
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_absolute_percentage_error

    evaluator = Evaluator(
        {
            "mse": mean_squared_error,
            "mae": mean_absolute_error,
            "mape": mean_absolute_percentage_error,
        }
    )
    ```

    The advantage of this is that this can easily be applied to each input dataset you use, so evaluations
    can be replicated with ease across datasets, so all metrics are directly available.
    """

    def __init__(
        self,
        functions: Dict[str, Callable[[Data, Data], Union[float, int, str, bool]]],
    ) -> None:
        """Evaluates a Model on the given Functions.

        Args:
            functions (Dict[str, Callable[[Data, Data], Union[float, int, str, bool]]]):
                A dict of metric names and metric functions. We expect each to return
                a float, int, string, or boolean value. We provide the arguments as follows:
                    - first, all labeled data, e.g. y_true.
                    - second, all predicted data, e.g. y_pred.
        """
        self.functions = functions

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
        if pipeline is not None:
            data = pipeline.transform(data)

        predictions = model.transform(data).get_all(model.outputs)
        output_data = data.get_all(model.outputs)

        return {name: func(*output_data, *predictions) for name, func in self.functions.items()}

    def __str__(self, indents: int = 0) -> str:
        """Create string representation of this Evaluator.

        Args:
            indents (int, optional): The number of preceding tabs. Defaults to 0.

        Returns:
            str: A string representation of this Evaluator.
        """
        tabs = "\t" * indents
        return "\n".join([f"{tabs}{name}: {func.__name__}" for name, func in self.functions.items()])
