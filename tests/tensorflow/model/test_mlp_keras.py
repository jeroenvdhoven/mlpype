import tensorflow as tf

from mlpype.tensorflow.model.mlp_keras import MLPKeras


def test_mlp_keras():
    n_hidden = 2
    output_size = 3
    size_hidden = 20
    hidden_act = "relu"

    model = MLPKeras(output_size, n_hidden, size_hidden, hidden_act)
    assert len(model._layers) == n_hidden + 1

    data = tf.random.normal((5, 17))
    result = model.call(data)
    assert result.shape == (5, output_size)
