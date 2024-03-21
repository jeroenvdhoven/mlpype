from typing import Any, List, Type, Union

import numpy as np
from pydantic import create_model

from mlpype.base.pipeline.type_checker import DataModel, TypeChecker


class NumpyData(DataModel):
    data: list

    """An object that can be converted to a numpy array."""

    def convert(self) -> np.ndarray:
        """Converts this object to a numpy array.

        Returns:
            np.ndarray: The numpy array contained by this object.
        """
        return np.array(self.data)

    @classmethod
    def to_model(cls, data: np.ndarray) -> "NumpyData":
        """Converts a numpy array to a NumpyData model, which can be serialised.

        Args:
            data (np.ndarray): A numpy array to serialise.

        Returns:
            NumpyData: A serialisable version of the array.
        """
        return NumpyData(data=data.tolist())


class NumpyTypeChecker(TypeChecker[np.ndarray]):
    def __init__(self) -> None:
        """Type checker for numpy.

        Checks if the incoming data is of the correct non-first dimensions and dtype.
        """
        super().__init__()
        self.dims: tuple = tuple()
        self.dtype: Union[type, None] = None

    def fit(self, data: np.ndarray) -> "NumpyTypeChecker":
        """Fit this Numpy TypeChecker to the given data.

        Args:
            data (np.ndarray): The data to fit.

        Returns:
            NumpyTypeChecker: self.
        """
        self.dims = data.shape[1:]
        self.dtype = self._convert_dtype(data.dtype)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Checks if the given data fits the specifications this TypeChecker was fitted for.

        Args:
            data (np.ndarray):  The data to check.

        Returns:
            np.ndarray: data, if the data fits the specifications. Otherwise, an assertion error is thrown.
        """
        assert self.dtype is not None, "Please fit pipeline first"
        assert isinstance(data, np.ndarray), "Please provide a numpy array!"
        assert data.shape[1:] == self.dims, f"Dimensions of numpy arrays do not add up: {data.shape[1:]} vs {self.dims}"

        converted_type = self._convert_dtype(data.dtype)
        assert converted_type == self.dtype, f"Dtype of data does not add up: {converted_type} vs {self.dtype}"
        return data

    def _convert_dtype(self, dtype: np.dtype) -> type:
        dtype_name = dtype.name
        if "int" in dtype_name:
            return int
        elif "float" in dtype_name:
            return float
        elif "bool" in dtype_name:
            return bool
        else:
            return str

    def get_pydantic_type(self) -> Type[NumpyData]:
        """Creates a Pydantic model for this data to handle serialisation/deserialisation.

        Returns:
            Type[NumpyData]: A NumpyData model that fits the data this wat fitted on.
        """
        base_iter: type = List[self.dtype]  # type: ignore

        for _ in range(len(self.dims)):
            base_iter = List[base_iter]  # type: ignore

        model = create_model("NumpyData", data=(base_iter, ...), __base__=NumpyData)

        return model

    @classmethod
    def supports_object(cls, obj: Any) -> bool:
        """Returns True if the object is a numpy array.

        Args:
            obj (Any): The object to check.

        Returns:
            bool: True if the given object is a numpy array, False otherwise.
        """
        return isinstance(obj, np.ndarray)
