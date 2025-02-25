"""Provides an implementation of the XGBoost model for mlpype."""
from xgboost.sklearn import XGBClassifier, XGBRanker, XGBRegressor

from .xgboost_model import XGBModel

XGBClassifierModel = XGBModel.class_from_sklearn_model_class(XGBClassifier)
XGBRegressorModel = XGBModel.class_from_sklearn_model_class(XGBRegressor)
XGBRankerModel = XGBModel.class_from_sklearn_model_class(XGBRanker)


__all__ = [
    "XGBClassifierModel",
    "XGBRegressorModel",
    "XGBRankerModel",
]
