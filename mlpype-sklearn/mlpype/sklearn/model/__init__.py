"""Provides classes for sklearn models.

For sklearn models not already configured here, you can use the SklearnModel class to quickly incorporate your model.


.. automodule:: mlpype.sklearn.model
   :members:
   :undoc-members:
   :show-inheritance:
"""

#  This weird import setup is required since sklearn.utils seems to want to refer to mlpype.sklearn.utils
import importlib
import sys

from .sklearn_base_type import SklearnModelBaseType
from .sklearn_model import SklearnModel

# from sklearn.linear_model import LinearRegression, LogisticRegression


utils = importlib.import_module("sklearn.utils", package=None)

current_module = sys.modules[__name__]
classes = utils.all_estimators(["classifier", "regressor"])
# classes = [LinearRegression, LogisticRegression]

dynamic_classes = []
for name, klass in classes:
    new_name = f"{name}Model"
    dynamic_class = SklearnModel.class_from_sklearn_model_class(klass)
    dynamic_classes.append(new_name)
    setattr(current_module, new_name, dynamic_class)

    # Update doc for dynamic class so sphinx can auto generate docs
#     __doc__ += f"""

# .. autoclass:: mlpype.sklearn.model.{new_name}
#     :members:

# """
__all__ = ["SklearnModel", "SklearnModelBaseType", *dynamic_classes]
