from typing import Any, Dict, List, Optional, Type

from mlpype.base.data import DataSet
from mlpype.base.pipeline import Operator


class Pipe:
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
        """A single step in a Pipeline.

        Used to fit an Operator to data and apply transformations to subsequent DataSets.

        Args:
            name (str): The name of the Pipe.
            operator (Type[Operator]): The Operator class that this Pipe should use.
            inputs (List[str]): A list of input dataset names used by this Pipe.
            outputs (List[str]): A list of output dataset names used by this Pipe.
            kw_args (Optional[Dict[str, Any]]): keyword arguments to initialise the Operator.
            fit_inputs: (Optional[List[str]]): optional additional arguments to fit().
                Will not be used in transform().
            skip_on_inference (Optional[bool]): Flag indicating if this step should be skipped
                at inference time. Useful to pre-process response variables in the pipeline.
        """
        assert "__" not in name, "Pipe names cannot contain the string `__`"
        if fit_inputs is None:
            fit_inputs = []
        if kw_args is None:
            kw_args = {}

        self.name = name
        self.operator_class = operator
        self.args = kw_args
        self.operator = operator(**kw_args)
        self.inputs = inputs
        self.outputs = outputs
        self.fit_inputs = fit_inputs
        self.skip_on_inference = skip_on_inference

    def fit(self, data: DataSet) -> "Pipe":
        """Fits the Pipe to the given DataSet.

        Args:
            data (DataSet): The DataSet to use in fitting.

        Returns:
            Pipe: This object.
        """
        self.operator.fit(*data.get_all(self.inputs), *data.get_all(self.fit_inputs))
        return self

    def transform(self, data: DataSet, is_inference: bool = False) -> DataSet:
        """Transforms the given data using this Pipe.

        This Pipe should be fitted first using fit().

        Args:
            data (DataSet): The DataSet to use in transforming.
            is_inference (Optional[bool]): Flag indicating if we're in inference
                mode for this transformation. We'll skip this step if
                skip_on_inference was set to True.

        Returns:
            DataSet: The transformed Data.
        """
        # skip this step if we're in inference mode and this Pipe is marked as such.
        if is_inference and self.skip_on_inference:
            return data

        transformed = self.operator.transform(*data.get_all(self.inputs))
        if len(self.outputs) < 2:
            transformed = [transformed]
        result = data.copy()
        result.set_all(self.outputs, transformed)
        return result

    def inverse_transform(self, data: DataSet, is_inference: bool = False) -> DataSet:
        """Inverse transforms the DataSet using this Pipe.

        Note that this is automatically done in reverse: the inverse steps
        will be applied from back to front.

        We do not inverse transform if not all outputs (the inputs for the reverse) are present.
        It is up to the user to make sure the inverse transformations
        work if used on a partial dataset (e.g. only the output data).
        In these cases, we return the `data` arg.

        Args:
            data (DataSet): The DataSet to use in inverse transforming.
            is_inference (Optional[bool]): Flag indicating if we're in inference
                mode for this inverse transformation. We'll skip this step if
                skip_on_inference was set to True.

        Returns:
            DataSet: The inverse transformed DataSet
        """
        # skip this step if we're in inference mode and this Pipe is marked as such.
        if is_inference and self.skip_on_inference:
            return data

        # We do not inverse transform if not all outputs (the inputs for the inverse) are present.
        # It is up to the user to make sure the inverse transformations
        # work if used on a partial dataset (e.g. only the output data).
        if not all([i in data for i in self.outputs]):
            return data
        inverse = self.operator.inverse_transform(*data.get_all(self.outputs))
        if len(self.inputs) < 2:
            inverse = [inverse]
        result = data.copy()
        result.set_all(self.inputs, inverse)
        return result

    def reinitialise(self, args: Dict[str, Any]) -> None:
        """Re-initialises this Pipe's Operator given a dict of arguments.

        Args:
            args (Dict[str, Any]): The dictionary of arguments to use in re-initialisation.
        """
        self.operator = self.operator_class(**args)

    def copy(self, args: Optional[Dict[str, Any]] = None) -> "Pipe":
        """Create a copy of this Pipe's Operator given a dict of arguments.

        Args:
            args (Optional[Dict[str, Any]]): The dictionary of arguments to use in re-initialisation.
                If set to None, we'll use the same arguments as before.

        Returns:
            A new copy of this Pipe, with a new Operator object.
        """
        if args is None:
            args = self.args

        return Pipe(
            self.name,
            self.operator_class,
            inputs=self.inputs,
            outputs=self.outputs,
            kw_args=args,
            fit_inputs=self.fit_inputs,
            skip_on_inference=self.skip_on_inference,
        )

    def __str__(self) -> str:
        """Create string representation of this Pipe.

        Returns:
            str: A string representation of this Pipe.
        """
        fit_section = f" (+ {self.fit_inputs})" if len(self.fit_inputs) > 0 else ""
        input_output_section = f"{self.inputs}{fit_section} -> {self.outputs}"

        return f"Pipe `{self.name}`, {input_output_section}: {self.operator}"
