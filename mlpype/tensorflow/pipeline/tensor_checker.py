from typing import Any, List, Type, Union

import tensorflow as tf
from pydantic import create_model
from tensorflow import DType, Tensor  # type: ignore

from mlpype.base.pipeline.type_checker import DataModel, TypeChecker


class TensorflowData(DataModel):
    data: list

    """An object that can be converted to a Tensorflow tensor."""

    def convert(self) -> Tensor:
        """Converts this object to a Tensorflow tensor.

        Returns:
            Tensor: The Tensorflow tensor contained by this object.
        """
        return tf.convert_to_tensor(self.data)  # type: ignore

    @classmethod
    def to_model(cls, data: Tensor) -> "TensorflowData":
        """Converts a Tensorflow tensor to a TensorflowData model, which can be serialised.

        Args:
            data (Tensor): A Tensorflow tensor to serialise.

        Returns:
            TensorflowData: A serialisable version of the tensor.
        """
        return TensorflowData(data=data.numpy().tolist())


class TensorflowTypeChecker(TypeChecker[Tensor]):
    def __init__(self) -> None:
        """Type checker for Tensorflow.

        Checks if the incoming data is of the correct non-first dimensions and dtype.
        """
        super().__init__()
        self.dims: tuple = tuple()
        self.dtype: Union[type, None] = None

    def fit(self, data: Tensor) -> "TensorflowTypeChecker":
        """Fit this Tensorflow TypeChecker to the given data.

        Args:
            data (Tensor): The data to fit.

        Returns:
            TensorflowTypeChecker: self.
        """
        self.dims = data.shape[1:]
        self.dtype = self._convert_dtype(data.dtype)
        return self

    def transform(self, data: Tensor) -> Tensor:
        """Checks if the given data fits the specifications this TypeChecker was fitted for.

        Args:
            data (Tensor):  The data to check.

        Returns:
            Tensor: data, if the data fits the specifications. Otherwise, an assertion error is thrown.
        """
        assert self.dtype is not None, "Please fit pipeline first"
        assert isinstance(data, Tensor), "Please provide a Tensorflow tensor!"
        assert (
            data.shape[1:] == self.dims
        ), f"Dimensions of Tensorflow tensors do not add up: {data.shape[1:]} vs {self.dims}"

        converted_type = self._convert_dtype(data.dtype)
        assert converted_type == self.dtype, f"Dtype of data does not add up: {converted_type} vs {self.dtype}"
        return data

    def _convert_dtype(self, dtype: DType) -> type:
        dtype_name = str(dtype)
        if "int" in dtype_name:
            return int
        elif "float" in dtype_name:
            return float
        elif "bool" in dtype_name:
            return bool
        else:
            return str

    def get_pydantic_type(self) -> Type[TensorflowData]:
        """Creates a Pydantic model for this data to handle serialisation/deserialisation.

        Returns:
            Type[TensorflowData]: A TensorflowData model that fits the data this wat fitted on.
        """
        base_iter: type = List[self.dtype]  # type: ignore

        for _ in range(len(self.dims)):
            base_iter = List[base_iter]  # type: ignore

        model = create_model("TensorflowData", data=(base_iter, ...), __base__=TensorflowData)

        return model

    @classmethod
    def supports_object(cls, obj: Any) -> bool:
        """Returns True if the object is a Tensorflow Tensor.

        Args:
            obj (Any): The object to check.

        Returns:
            bool: True if the given object is a Tensor, False otherwise.
        """
        return isinstance(obj, Tensor)
