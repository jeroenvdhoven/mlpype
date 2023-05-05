import tensorflow as tf
from numpy.testing import assert_array_equal

from mlpype.tensorflow.data.tensor_source import TensorSource


def test_TensorSource():
    tensor = tf.convert_to_tensor([1, 2, 3, 4])

    source = TensorSource(tensor)

    assert_array_equal(tensor.numpy(), source.read().numpy())
