import numpy as np
import tensorflow as tf
from pydantic import ValidationError
from pytest import mark

from mlpype.tensorflow.pipeline.tensor_checker import TensorflowData, TensorflowTypeChecker
from tests.utils import pytest_assert


class Test_TensorflowData:
    def test_convert(self):
        model = TensorflowData(data=[[1, 2, 3], [4, 5, 6], [6, 7, 8]])
        expected = tf.convert_to_tensor([[1, 2, 3], [4, 5, 6], [6, 7, 8]])

        result = model.convert()

        assert isinstance(result, tf.Tensor)
        np.testing.assert_array_equal(expected.numpy(), result.numpy())

    def test_to_model(self):
        array = tf.convert_to_tensor([[1, 2, 3], [2, 3, 4]])
        result = TensorflowData.to_model(array)

        expected = TensorflowData(
            data=[[1, 2, 3], [2, 3, 4]],
        )

        assert expected == result


class Test_TensorflowTypeChecker:
    def test_fit(self):
        type_checker = TensorflowTypeChecker()
        array = tf.convert_to_tensor([[1, 2, 3], [3, 4, 5]])
        assert array.shape[0] != array.shape[1]

        type_checker.fit(array)

        assert type_checker.dims == (3,)
        assert type_checker.dtype == int

    def test_transform(self):
        type_checker = TensorflowTypeChecker()

        array = tf.convert_to_tensor([[1, 2, 3], [3, 4, 5]])
        type_checker.fit(array)

        # works
        result = type_checker.transform(array)

        # errors #
        # other type of object
        with pytest_assert(AssertionError, "Please provide a Tensorflow tensor!"):
            type_checker.transform([])

        # other shape
        other_shape = tf.convert_to_tensor([[1, 2], [3, 4]])
        with pytest_assert(
            AssertionError,
            f"Dimensions of Tensorflow tensors do not add up: {other_shape.shape[1:]} vs {array.shape[1:]}",
        ):
            type_checker.transform(other_shape)

        # other type
        other_type = tf.convert_to_tensor([[1, 2, 3], [3, 4, 5.0]])
        with pytest_assert(AssertionError, f"Dtype of data does not add up: {float} vs {int}"):
            type_checker.transform(other_type)

    @mark.parametrize(
        ["input_array", "expected"],
        [
            [tf.convert_to_tensor([1, 2, 3]), int],
            [tf.convert_to_tensor([1.0, 2, 3]), float],
            [tf.convert_to_tensor([False, True]), bool],
            [tf.convert_to_tensor(["1", "2", "3"]), str],
        ],
    )
    def test_convert_dtype(self, input_array: tf.Tensor, expected: type):
        type_checker = TensorflowTypeChecker()

        result = type_checker._convert_dtype(input_array.dtype)
        assert result == expected

    def test_get_pydantic_types(self):
        name = "tf_data"
        type_checker = TensorflowTypeChecker(name)
        array = tf.convert_to_tensor([[1, 2, 3], [7, 6, 5]])
        type_checker.fit(array)

        TensorflowSpecificType = type_checker.get_pydantic_type()
        assert TensorflowSpecificType.__name__ == f"TensorflowData[{name}]"

        # succeed
        TensorflowSpecificType(data=[[1, 2, 3], [9, 0, 3]])

        # also succeed
        data = [[1, 2, 3], [9, 0, 3.0]]
        result = TensorflowSpecificType(data=data).convert()

        assert isinstance(result, tf.Tensor)
        np.testing.assert_array_equal(tf.convert_to_tensor(data).numpy(), result.numpy())

    def test_get_pydantic_types_strict_on_bool(self):
        type_checker = TensorflowTypeChecker()
        array = tf.convert_to_tensor([[True, False]])
        type_checker.fit(array)

        TensorflowSpecificType = type_checker.get_pydantic_type()

        # succeed
        TensorflowSpecificType(data=[[False, False]])

        # fails
        with pytest_assert(ValidationError):
            TensorflowSpecificType(data=[[1, 2, 3]])

    @mark.parametrize(["obj", "expected"], [[[], False], [1, False], [tf.convert_to_tensor([1.0, 2.0]), True]])
    def test_supports_object(self, obj, expected: bool):
        assert TensorflowTypeChecker.supports_object(obj) == expected
