"""Provides tools to evaluate Spark models in mlpype."""
from typing import Dict, List, Optional, Union

from pyspark.ml.evaluation import JavaEvaluator
from pyspark.sql import DataFrame as SparkDataFrame

from mlpype.base.data.dataset import DataSet
from mlpype.base.evaluate.base_evaluator import BaseEvaluator
from mlpype.base.pipeline import Pipeline
from mlpype.spark.model import SparkModel


class SparkEvaluator(BaseEvaluator):
    """Used to evaluate Spark models in mlpype."""

    def __init__(
        self,
        evaluator: JavaEvaluator,
        metrics: List[str],
        set_cache: bool = False,
    ) -> None:
        """Used to evaluate Spark models in mlpype.

        Args:
            evaluator (JavaEvaluator): One of the Evaluators from pyspark, e.g.
                psypark.ml.evaluation.RegressionEvaluator
            metrics (List[str]): The list of metrics to retrieve using the Evaluator.
                See the documentation of the one you use to know which are available.
            set_cache (bool, optional): If True, the predictions will be cached just before
                evaluation. This can help speed up the evaluation. Defaults to False.
        """
        super().__init__()
        self.evaluator = evaluator
        self.metrics = metrics
        self.set_cache = set_cache

    def evaluate(
        self, model: SparkModel, data: DataSet, pipeline: Optional[Pipeline] = None
    ) -> Dict[str, Union[str, float, int, str, bool]]:
        """Evaluate the given model on the given dataset.

        We assume the model does not need to be transformed anymore if pipeline is None.

        Args:
            model (SparkModel): The SparkModel to evaluate.
            data (DataSet): The Dataset to use to evaluate the model.
            pipeline (Optional[Pipeline]): If not None, this will be used to transform `data` first.

        Returns:
            Dict[str, Union[str, float, int, str, bool]]: A dictionary of metric_name-value pairs. The result
                of the evaluation.
        """
        if pipeline is not None:
            data = pipeline.transform(data)

        predictions = model.transform(data).get_all(model.outputs)

        assert len(predictions) == 1, "Expect exactly 1 output from a SparkModel."
        preds: SparkDataFrame = predictions[0]
        if self.set_cache:
            preds = preds.cache()

        result = {}
        for metric in self.metrics:
            value = self.evaluator.setMetricName(metric).evaluate(preds)  # type: ignore
            result[metric] = value
        return result

    def __str__(self, indents: int = 0) -> str:
        """Generates a string representation of this SparkEvaluator.

        Args:
            indents (int, optional): The number of tabs to preceed this string
                representation. Defaults to 0.

        Returns:
            str: A string representation of this SparkEvaluator.
        """
        tabs = "\t" * indents
        metrics = ", ".join(self.metrics)
        return f"{tabs}{type(self.evaluator).__name__}: {metrics}"
