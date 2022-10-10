from pathlib import Path
from typing import Any, Type

from xgboost.sklearn import XGBClassifier

from pype.sklearn.model import SklearnModel


class XGBClassifierModel(SklearnModel[XGBClassifier]):
    XGB_MODEL_FILE = "model.txt"

    def _init_model(self, args: dict[str, Any]) -> XGBClassifier:
        return XGBClassifier(**args)

    def _save(self, folder: Path) -> None:
        assert isinstance(self.model, XGBClassifier)
        self.model.save_model(folder / self.XGB_MODEL_FILE)

    @classmethod
    def _load(cls: Type["XGBClassifierModel"], folder: Path, inputs: list[str], outputs: list[str]) -> "SklearnModel":
        model = XGBClassifier()
        model.load_model(folder / cls.XGB_MODEL_FILE)
        return cls(inputs=inputs, outputs=outputs, model=model, seed=1)