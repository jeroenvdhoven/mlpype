"""Provides classes for sklearn models.

For sklearn models not already configured here, you can use the SklearnModel class to quickly incorporate your model.

.. automodule:: mlpype.sklearn.models
   :members:
   :undoc-members:
   :show-inheritance:
"""

import sys

from sklearn.utils import all_estimators

from .sklearn_base_type import SklearnModelBaseType
from .sklearn_model import SklearnModel

current_module = sys.modules[__name__]
classes = all_estimators(["classifier", "regressor"])

dynamic_classes = []
for name, klass in classes:
    new_name = f"{name}Model"
    dynamic_class = SklearnModel.class_from_sklearn_model_class(klass)
    dynamic_classes.append(new_name)
    setattr(current_module, new_name, dynamic_class)


__all__ = ["SklearnModel", "SklearnModelBaseType", *dynamic_classes]
