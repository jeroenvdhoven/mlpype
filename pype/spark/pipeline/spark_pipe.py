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
            raise ValueError(f"In a SparkPipe, the operator must be a Transformer or Estimator. Got: {type(op)}")
        return self

    def transform(self, data: DataSet) -> DataSet:
        """Transforms the given data using this Pipe.

        This Pipe should be fitted first using fit().
        This is version of Pipe is changed to work with Spark's API: transformation is done using the
        fitted object instead of the Operator directly.

        Args:
            data (DataSet): The DataSet to use in transforming.

        Returns:
            DataSet: The transformed Data.
        """
        transformed = self.fitted.transform(*data.get_all(self.inputs))
        if len(self.outputs) < 2:
            transformed = [transformed]
        result = data.copy()
        result.set_all(self.outputs, transformed)
        return result

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Sets the state of this object from a dictionary.

        Used by pickle to properly prepare the "fitted" and "operator" fields as None's.

        Args:
            state (dict[str, Any]): _description_
        """
        state["fitted"] = None
        state["operator"] = None
        self.__dict__ = state

    def __getstate__(self) -> dict[str, Any]:
        """Gets the state of this object, excluding any Spark objects.

        This is to make sure serialisation works as expected.

        Returns:
            dict[str, Any]: A dict representation of this object.
        """
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
