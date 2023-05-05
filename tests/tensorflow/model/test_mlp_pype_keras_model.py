import tensorflow as tf
from keras.losses import MeanAbsoluteError
from keras.optimizers import Nadam

from mlpype.base.data.dataset import DataSet
from mlpype.tensorflow.model.mlp_pype_model import MLPPypeModel


def test_mlp_mlpype_keras_model():
    n = 1000
    x = tf.random.normal((n, 4))
    y = x[:, 0] + x[:, 1] * -0.4 + x[:, 2] * 2 + x[:, 3] * -0.4

    model = MLPPypeModel(
        ["x"],
        ["y"],
        epochs=10,
        batch_size=25,
        optimizer_class=Nadam,
        learning_rate=0.01,
        loss=MeanAbsoluteError(),
        output_size=1,
        n_layers=1,
        layer_size=5,
    )
    dataset = DataSet(x=x, y=y)

    old_predictions = model.transform(dataset)["y"]
    model.fit(dataset)

    fit_predictions = model.transform(dataset)["y"]

    old_error = MeanAbsoluteError()(y, old_predictions)
    fit_error = MeanAbsoluteError()(y, fit_predictions)

    assert fit_error < old_error
