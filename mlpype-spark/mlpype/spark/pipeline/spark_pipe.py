from typing import Any, Dict, List, Optional, Type, Union

from pyspark.ml import Estimator, Transformer

from mlpype.base.data.dataset import DataSet
from mlpype.base.pipeline.operator import Operator
from mlpype.base.pipeline.pipe import Pipe


class SparkPipe(Pipe):
    def __init__(
        self,
        name: str,
        operator: Type[Operator],
        inputs: List[str],
        outputs: List[str],
        kw_args: Optional[Dict[str, Any]] = None,
        fit_inputs: Optional[List[str]] = None,
        skip_on_inference: bool = False,
    ) -> None:
        """Same init as Pipe."""
        super().__init__(name, operator, inputs, outputs, kw_args, fit_inputs, skip_on_inference)
        self.fitted = None

    def fit(self, data: DataSet) -> "SparkPipe":
        """Fits the SparkPipe to the given DataSet.

        Args:
            data (DataSet): The DataSet to use in fitting.

        Returns:
            SparkPipe: This object.
        """
        op: Union[Transformer, Estimator] = self.operator

        if isinstance(op, Transformer):
            self.fitted = op
        elif isinstance(op, Estimator):
            self.fitted = op.fit(*data.get_all(self.inputs), *data.get_all(self.fit_inputs))
        else:
            raise ValueError(f"In a SparkPipe, the operator must be a Transformer or Estimator. Got: {type(op)}")
        return self

    def transform(self, data: DataSet, is_inference: bool = False) -> DataSet:
        """Transforms the given data using this Pipe.

        This Pipe should be fitted first using fit().
        This is version of Pipe is changed to work with Spark's API: transformation is done using the
        fitted object instead of the Operator directly.

        Args:
            data (DataSet): The DataSet to use in transforming.
            is_inference (Optional[bool]): Flag indicating if we're in inference
                mode for this transformation. We'll skip this step if
                skip_on_inference was set to True.

        Returns:
            DataSet: The transformed Data.
        """
        assert self.fitted is not None, "Make sure you fit the pipeline before transforming."

        # skip this step if we're in inference mode and this Pipe is marked as such.
        if is_inference and self.skip_on_inference:
            return data

        transformed = self.fitted.transform(*data.get_all(self.inputs))
        if len(self.outputs) < 2:
            transformed = [transformed]
        result = data.copy()
        result.set_all(self.outputs, transformed)
        return result

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Sets the state of this object from a dictionary.

        Used by pickle to properly prepare the "fitted" and "operator" fields as None's.

        Args:
            state (Dict[str, Any]): The set of parameters that this Pipe should be set to.
        """
        state["fitted"] = None
        state["operator"] = None
        self.__dict__ = state

    def __getstate__(self) -> Dict[str, Any]:
        """Gets the state of this object, excluding any Spark objects.

        This is to make sure serialisation works as expected.

        Returns:
            Dict[str, Any]: A dict representation of this object.
        """
        dct = self.__dict__.copy()
        if "fitted" in dct:
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

        postfix = "unfitted" if self.fitted is None else str(self.fitted)

        return f"Pipe `{self.name}`, {input_output_section}: {postfix}"
