from typing import Any

from keras import Model
from keras.layers import Dense
from tensorflow import Tensor  # type: ignore


class MLPKeras(Model):
    def __init__(
        self,
        output_size: int,
        n_layers: int,
        layer_size: int,
        activation: str = None,
        output_activation: str = None,
        *args: Any,
        **kwargs: Any,
    ):
        """A simple model showing how to use a MLP from Keras with mlpype.

        Args:
            output_size (int): The dimensions of the output
            n_layers (int): The number of hidden layers
            layer_size (int): The dimensions of hidden layers.
            activation (str, optional): The activation function of hidden layers. Defaults to None.
            output_activation (str, optional): The output activation function. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        assert n_layers >= 0, "Need at least 1 layer in the model."

        activations = [activation for _ in range(n_layers)] + [output_activation]
        sizes = [layer_size for _ in range(n_layers)] + [output_size]
        self._layers = [Dense(size, activation=act) for act, size in zip(activations, sizes)]

    def call(self, inputs: Tensor) -> Tensor:
        """Standard Keras model API call."""
        for layer in self._layers:
            inputs = layer(inputs)
        return inputs
