from datetime import datetime
from typing import Dict, List, Type, Union

import pandas as pd
from pandas.api import types as pandas_types
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
)
from pandas.testing import assert_frame_equal
from pydantic import create_model
from pytest import fixture

from pype.sklearn.pipeline.pandas_type_checker import PandasData, PandasTypeChecker
from tests.utils import pytest_assert


class Test_PandasData:
    @fixture
    def model_class(self) -> Type[PandasData]:
        return create_model(
            "PandasData",
            x=(Union[List[int], Dict[Union[str, int], int]], ...),
            y=(Union[List[str], Dict[Union[str, int], str]], ...),
            __base__=PandasData,
        )

    def test_convert(self, model_class: Type[PandasData]):
        model = model_class(x=[1, 2, 3, 4], y=["a", "b", "c", "e"])

        expected = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": ["a", "b", "c", "e"],
            }
        )

        result = model.convert()

        assert_frame_equal(expected, result)

    def test_to_model(self, model_class: Type[PandasData]):
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": ["a", "b", "c", "e"],
            }
        )
        result = model_class.to_model(df)

        expected = model_class(
            x=[1, 2, 3, 4],
            y=["a", "b", "c", "e"],
        )

        assert expected == result


class Test_PandasTypeChecker:
    def test_fit(self):
        type_checker = PandasTypeChecker()
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": ["a", "b", "c", "e"],
            }
        )

        type_checker.fit(df)
        expected_dict = {
            "x": (int, is_integer_dtype),
            "y": (str, is_object_dtype),
        }
        assert type_checker.raw_types == expected_dict

    def test_transform(self):
        type_checker = PandasTypeChecker()
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": ["a", "b", "c", "e"],
            }
        )

        type_checker.fit(df)
        # works
        result = type_checker.transform(df)

        # errors #
        # other type of object
        with pytest_assert(AssertionError, "Please provide a pandas DataFrame!"):
            type_checker.transform([])

        # missing columns
        missing_df = pd.DataFrame({"x": [1, 2, 3]})  # missing
        with pytest_assert(AssertionError, "Not all columns are present."):
            type_checker.transform(missing_df)

        other_type = df.copy()
        other_type["x"] = other_type["x"].astype(str)

        with pytest_assert(AssertionError, "Dtypes did not match up for col x."):
            type_checker.transform(other_type)

    def test_transform_reduces_columns(self):
        type_checker = PandasTypeChecker()
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": ["a", "b", "c", "e"],
            }
        )

        type_checker.fit(df)
        # works
        extra_col_df = df.copy()
        extra_col_df["z"] = 1
        result = type_checker.transform(extra_col_df)

        assert "z" not in df.columns

    def test_convert_raw_types(self):
        type_checker = PandasTypeChecker()
        df = pd.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [1, 2, 3.0],
                "z": ["1", "2", "3"],
                "a": [True, False, True],
                "b": [1, "1", 1],
                "c": [datetime(2018, 1, 1, 1), datetime(2018, 1, 1, 0), datetime(2018, 1, 1, 2)],
            }
        )

        result = type_checker._convert_raw_types(dict(df.dtypes))
        expected = {
            "x": (int, pandas_types.is_integer_dtype),
            "y": (float, pandas_types.is_float_dtype),
            "z": (str, pandas_types.is_object_dtype),
            "a": (bool, pandas_types.is_bool_dtype),
            "b": (str, pandas_types.is_object_dtype),
            "c": (datetime, pandas_types.is_datetime64_any_dtype),
        }
        assert result == expected

    def test_get_pydantic_types(self):
        type_checker = PandasTypeChecker()
        df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": ["a", "b", "c", "e"],
            }
        )
        type_checker.fit(df.copy())

        PandasSpecificType = type_checker.get_pydantic_type()

        # succeed
        result = PandasSpecificType(x=[1, 2, 3, 4], y=["a", "b", "c", "e"]).convert()

        assert_frame_equal(df, result)

    # def test_get_pydantic_types_strict_on_bool(self):
    #     type_checker = PandasTypeChecker()
    #     array = np.array([
    #         [True, False]
    #     ])
    #     type_checker.fit(array)

    #     PandasSpecificType = type_checker.get_pydantic_type()

    #     # succeed
    #     PandasSpecificType(data=[[False, False]])

    #     # fails
    #     with pytest_assert(ValidationError):
    #         PandasSpecificType(data=[[1, 2 ,3]])
