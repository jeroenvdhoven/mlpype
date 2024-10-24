"""Provides classes for sklearn models.

For sklearn models not already configured here, you can use the SklearnModel class to quickly incorporate your model.
We have already integrated classifier and regressor models from sklearn. You can import them like:
```python
from mlpype.sklearn.model import <sklearn name>Model
```
"""

#  This weird import setup is required since sklearn.utils seems to want to refer to mlpype.sklearn.utils
import importlib
import sys

from .sklearn_base_type import SklearnModelBaseType
from .sklearn_model import SklearnModel

# from sklearn.linear_model import LinearRegression, LogisticRegression


__utils = importlib.import_module("sklearn.utils", package=None)
__classes = __utils.all_estimators(["classifier", "regressor"])
# classes = [LinearRegression, LogisticRegression]

__dynamic_classes = []
for name, klass in __classes:
    __new_name = f"{name}Model"
    __dynamic_class = SklearnModel.class_from_sklearn_model_class(klass)
    __dynamic_classes.append(__new_name)
    setattr(sys.modules[__name__], __new_name, __dynamic_class)

__all__ = ["SklearnModel", "SklearnModelBaseType", *__dynamic_classes]
