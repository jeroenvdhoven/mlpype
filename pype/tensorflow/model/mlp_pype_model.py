from typing import Any

from .keras_pype_model import KerasPypeModel
from .mlp_keras import MLPKeras


class MLPPypeModel(KerasPypeModel[MLPKeras]):
    """A Keras model integrated with Pype's APIs."""

    def _init_model(self, args: dict[str, Any]) -> MLPKeras:
        return MLPKeras(**args)
