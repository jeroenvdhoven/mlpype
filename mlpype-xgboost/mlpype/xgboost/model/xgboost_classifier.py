"""Provides an implementation of the XGBoost Classifier model for mlpype."""
from pathlib import Path
from typing import List, Type

from xgboost.sklearn import XGBClassifier

from mlpype.sklearn.model import SklearnModel


class XGBClassifierModel(SklearnModel[XGBClassifier]):
    """Provides an implementation of the XGBoost Classifier model for mlpype."""

    XGB_MODEL_FILE = "model.txt"

    def _save(self, folder: Path) -> None:
        assert isinstance(self.model, XGBClassifier)
        self.model.save_model(folder / self.XGB_MODEL_FILE)

    @classmethod
    def _load(cls: Type["XGBClassifierModel"], folder: Path, inputs: List[str], outputs: List[str]) -> "SklearnModel":
        model = XGBClassifier()
        model.load_model(folder / cls.XGB_MODEL_FILE)
        return cls(inputs=inputs, outputs=outputs, model=model, seed=1)
