from typing import Any, Dict, Type

from pype.base.data import DataSet
from pype.base.pipeline import Operator


class Pipe:
    def __init__(
        self,
        name: str,
        operator: Type[Operator],
        inputs: list[str],
        outputs: list[str],
        kw_args: dict[str, Any] | None = None,
        fit_inputs: list[str] | None = None,
    ) -> None:
        """A single step in a Pipeline.

        Used to fit an Operator to data and apply transformations to subsequent DataSets.

        Args:
            name (str): The name of the Pipe.
            operator (Type[Operator]): The Operator class that this Pipe should use.
            inputs (list[str]): A list of input dataset names used by this Pipe.
            outputs (list[str]): A list of output dataset names used by this Pipe.
            kw_args (dict[str, Any] | None): keyword arguments to initialise the Operator.
            fit_inputs: (list[str] | None): optional additional arguments to fit().
                Will not be used in transform().
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

    def fit(self, data: DataSet) -> "Pipe":
        """Fits the Pipe to the given DataSet.

        Args:
            data (DataSet): The DataSet to use in fitting.

        Returns:
            Pipe: This object.
        """
        self.operator.fit(*data.get_all(self.inputs), *data.get_all(self.fit_inputs))
        return self

    def transform(self, data: DataSet) -> DataSet:
        """Transforms the given data using this Pipe.

        This Pipe should be fitted first using fit().

        Args:
            data (DataSet): The DataSet to use in transforming.

        Returns:
            DataSet: The transformed Data.
        """
        transformed = self.operator.transform(*data.get_all(self.inputs))
        if len(self.outputs) < 2:
            transformed = [transformed]
        result = data.copy()
        result.set_all(self.outputs, transformed)
        return result

    def inverse_transform(self, data: DataSet) -> DataSet:
        """Inverse transforms the DataSet using this Pipe.

        Note that this is automatically done in reverse: the inverse steps
        will be applied from back to front.

        We do not inverse transform if not all outputs (the inputs for the reverse) are present.
        It is up to the user to make sure the inverse transformations
        work if used on a partial dataset (e.g. only the output data).
        In these cases, we return the `data` arg.

        Args:
            data (DataSet): The DataSet to use in inverse transforming.

        Returns:
            DataSet: The inverse transformed DataSet
        """
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

    def copy(self, args: Dict[str, Any] | None = None) -> "Pipe":
        """Create a copy of this Pipe's Operator given a dict of arguments.

        Args:
            args (Dict[str, Any] | None): The dictionary of arguments to use in re-initialisation.
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
        )
