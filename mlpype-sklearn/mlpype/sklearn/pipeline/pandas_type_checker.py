from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
)
from pydantic import create_model

from mlpype.base.pipeline.type_checker import DataModel, TypeChecker


class PandasData(DataModel):
    def convert(self) -> pd.DataFrame:
        """Converts this object to a pandas DataFrame.

        Returns:
            pd.DataFrame: The pandas DataFrame contained by this object.
        """
        return pd.DataFrame(data=self.__dict__)

    @classmethod
    def to_model(cls, data: pd.DataFrame) -> "PandasData":
        """Converts a pandas DataFrame to a PandasData model, which can be serialised.

        Args:
            data (pd.DataFrame): A pandas DataFrame to serialise.

        Returns:
            PandasData: A serialisable version of the DataFrame.
        """
        return cls(**{name: [str(i) for i in data[name].to_list()] for name in data.columns})


class PandasTypeChecker(TypeChecker[pd.DataFrame]):
    def fit(self, data: pd.DataFrame) -> "PandasTypeChecker":
        """Fit this PandasTypeChecker to the given data.

        Args:
            data (pd.DataFrame): The data to fit.

        Returns:
            PandasTypeChecker: self
        """
        self.raw_types = self._convert_raw_types(dict(data.dtypes))
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Checks if the given data fits the specifications this TypeChecker was fitted for.

        Args:
            data (pd.DataFrame): The data to check.

        Returns:
            pd.DataFrame: data, if the data fits the specifications. Otherwise, an assertion error is thrown.
        """
        assert isinstance(data, pd.DataFrame), "Please provide a pandas DataFrame!"
        colnames = list(self.raw_types.keys())
        assert np.all(np.isin(colnames, data.columns)), "Not all columns are present."

        data = data[colnames]

        for name, (_, checker) in self.raw_types.items():
            assert checker(data[name]), f"Dtypes did not match up for col {name}."
        return data

    def _convert_raw_types(self, types: Dict[str, type]) -> Dict[str, Tuple[type, Callable]]:
        return {name: self._convert_raw_type(type_) for name, type_ in types.items()}

    def _convert_raw_type(self, type_: type) -> Tuple[type, Callable]:
        str_type = str(type_)
        if "int" in str_type:
            return (int, is_integer_dtype)
        elif "float" in str_type:
            return (float, is_float_dtype)
        elif "bool" in str_type:
            return (bool, is_bool_dtype)
        elif "datetime" in str_type:
            return (datetime, is_datetime64_any_dtype)
        elif "str" in str_type:
            return (str, is_string_dtype)
        else:
            return (str, is_object_dtype)

    def get_pydantic_type(self) -> Type[PandasData]:
        """Creates a Pydantic model for this data to handle serialisation/deserialisation.

        Returns:
            Type[PandasData]: A PandasData model that fits the data this wat fitted on.
        """
        data_type = {
            name: (Union[List[dtype], Dict[str or int, dtype]], ...)  # type: ignore
            for name, (dtype, _) in self.raw_types.items()
        }

        model = create_model("PandasData", **data_type, __base__=PandasData)

        return model

    @classmethod
    def supports_object(cls, obj: Any) -> bool:
        """Returns True if the object is a pandas DataFrame.

        Args:
            obj (Any): The object to check.

        Returns:
            bool: True if the given object is a pandas DataFrame, False otherwise.
        """
        return isinstance(obj, pd.DataFrame)
