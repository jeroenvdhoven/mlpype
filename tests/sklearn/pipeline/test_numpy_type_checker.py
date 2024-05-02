import numpy as np
import pandas as pd
from pydantic import ValidationError
from pytest import mark

from mlpype.sklearn.pipeline.numpy_type_checker import NumpyData, NumpyTypeChecker
from tests.utils import pytest_assert


class Test_NumpyData:
    def test_convert(self):
        model = NumpyData(data=[[1, 2, 3], [4, 5, 6], [6, 7, 8]])
        expected = np.array([[1, 2, 3], [4, 5, 6], [6, 7, 8]])

        result = model.convert()

        np.testing.assert_array_equal(expected, result)

    def test_to_model(self):
        array = np.array([[1, 2, 3], [2, 3, 4]])
        result = NumpyData.to_model(array)

        expected = NumpyData(
            data=[[1, 2, 3], [2, 3, 4]],
        )

        assert expected == result


class Test_NumpyTypeChecker:
    def test_fit(self):
        type_checker = NumpyTypeChecker()
        array = np.array([[1, 2, 3], [3, 4, 5]])
        assert array.shape[0] != array.shape[1]

        type_checker.fit(array)

        assert type_checker.dims == (3,)
        assert type_checker.dtype == int

    def test_transform(self):
        type_checker = NumpyTypeChecker()

        array = np.array([[1, 2, 3], [3, 4, 5]])
        type_checker.fit(array)

        # works
        result = type_checker.transform(array)

        # errors #
        # other type of object
        with pytest_assert(AssertionError, "Please provide a numpy array!"):
            type_checker.transform([])

        # other shape
        other_shape = np.array([[1, 2], [3, 4]])
        with pytest_assert(
            AssertionError, f"Dimensions of numpy arrays do not add up: {other_shape.shape[1:]} vs {array.shape[1:]}"
        ):
            type_checker.transform(other_shape)

        # other type
        other_type = np.array([[1, 2, 3], [3, 4, 5.0]])
        with pytest_assert(AssertionError, f"Dtype of data does not add up: {float} vs {int}"):
            type_checker.transform(other_type)

    @mark.parametrize(
        ["input_array", "expected"],
        [
            [np.array([1, 2, 3]), int],
            [np.array([1.0, 2, 3]), float],
            [np.array([False, True]), bool],
            [np.array(["1", "2", "3"]), str],
        ],
    )
    def test_convert_dtype(self, input_array: np.ndarray, expected: type):
        type_checker = NumpyTypeChecker()

        result = type_checker._convert_dtype(input_array.dtype)
        assert result == expected

    def test_get_pydantic_types(self):
        type_checker = NumpyTypeChecker()
        array = np.array([[1, 2, 3], [7, 6, 5]])
        type_checker.fit(array)

        NumpySpecificType = type_checker.get_pydantic_type()

        # succeed
        NumpySpecificType(data=[[1, 2, 3], [9, 0, 3]])

        # also succeed
        data = [[1, 2, 3], [9, 0, 3.0]]
        result = NumpySpecificType(data=data).convert()

        np.testing.assert_array_equal(np.array(data), result)

    def test_get_pydantic_types_strict_on_bool(self):
        name = "dataset_na,e"
        type_checker = NumpyTypeChecker(name)
        array = np.array([[True, False]])
        type_checker.fit(array)

        NumpySpecificType = type_checker.get_pydantic_type()
        assert NumpySpecificType.__name__ == f"NumpyData[{name}]"

        # succeed
        NumpySpecificType(data=[[False, False]])

        # fails
        with pytest_assert(ValidationError):
            NumpySpecificType(data=[[1, 2, 3]])

    @mark.parametrize(
        ["obj", "expected"], [[[], False], [1, False], [pd.DataFrame({"a": [1]}), False], [np.array([1.0, 2.0]), True]]
    )
    def test_supports_object(self, obj, expected: bool):
        assert NumpyTypeChecker.supports_object(obj) == expected
