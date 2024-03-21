from typing import Any, Dict

from .keras_pype_model import KerasPypeModel
from .mlp_keras import MLPKeras


class MLPPypeModel(KerasPypeModel[MLPKeras]):
    """A Keras model integrated with mlpype's APIs."""

    def _init_model(self, args: Dict[str, Any]) -> MLPKeras:
        return MLPKeras(**args)
