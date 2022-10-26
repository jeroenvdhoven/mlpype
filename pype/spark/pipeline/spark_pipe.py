from typing import Any

from data.dataset import DataSet
from pyspark.ml import Estimator, Transformer

from pype.base.pipeline.pipe import Pipe


class SparkPipe(Pipe):
    def fit(self, data: DataSet) -> "SparkPipe":
        """Fits the SparkPipe to the given DataSet.

        Args:
            data (DataSet): The DataSet to use in fitting.

        Returns:
            SparkPipe: This object.
        """
        op: Transformer | Estimator = self.operator

        if isinstance(op, Transformer):
            self.fitted = op
        elif isinstance(op, Estimator):
            self.fitted = op.fit(*data.get_all(self.inputs), *data.get_all(self.fit_inputs))
        else:
            raise ValueError(f"In a SparkPipe, the operator must be a Transformer or Estimator.")
        return self

    def transform(self, data: DataSet) -> DataSet:
        transformed = self.fitted.transform(*data.get_all(self.inputs))
        if len(self.outputs) < 2:
            transformed = [transformed]
        result = data.copy()
        result.set_all(self.outputs, transformed)
        return result

    def __setstate__(self, state: dict[str, Any]):
        state["fitted"] = None
        state["operator"] = None
        self.__dict__ = state

    def __getstate__(self):
        dct = self.__dict__.copy()
        del dct["fitted"]
        del dct["operator"]
        return dct

    def __str__(self) -> str:
        """Create string representation of this Pipe.

        Returns:
            str: A string representation of this Pipe.
        """
        fit_section = f" (+ {self.fit_inputs})" if len(self.fit_inputs) > 0 else ""
        input_output_section = f"{self.inputs}{fit_section} -> {self.outputs}"

        return f"Pipe `{self.name}`, {input_output_section}: {self.fitted}"
