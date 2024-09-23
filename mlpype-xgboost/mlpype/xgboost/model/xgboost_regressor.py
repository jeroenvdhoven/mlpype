"""Provides an implementation of the XGBoost Regressor model for mlpype."""
from pathlib import Path
from typing import List, Type

from xgboost.sklearn import XGBRegressor

from mlpype.sklearn.model import SklearnModel


class XGBRegressorModel(SklearnModel[XGBRegressor]):
    """Provides an implementation of the XGBoost Regressor model for mlpype."""

    XGB_MODEL_FILE = "model.txt"

    def _save(self, folder: Path) -> None:
        assert isinstance(self.model, XGBRegressor)
        self.model.save_model(folder / self.XGB_MODEL_FILE)

    @classmethod
    def _load(cls: Type["XGBRegressorModel"], folder: Path, inputs: List[str], outputs: List[str]) -> "SklearnModel":
        model = XGBRegressor()
        model.load_model(folder / cls.XGB_MODEL_FILE)
        return cls(inputs=inputs, outputs=outputs, model=model, seed=1)
