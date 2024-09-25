"""Provides an implementation of the XGBoost models for mlpype."""
from pathlib import Path
from typing import List, Type, TypeVar

from xgboost.sklearn import XGBModel as BaseXGBModel

from mlpype.sklearn.model.sklearn_model import SklearnModel

T = TypeVar("T", bound=BaseXGBModel)


class XGBModel(SklearnModel[T]):
    """Provides an implementation of the XGBoost models for mlpype.

    This should be generic enough to support any XGBoost model. It extends the SklearnModel
    with specific XGBoost functionality for saving and loading models. If you want to
    use a specific (currently not already created xgboost model), just use the following example:

    ```python
    from mlpype.xgboost.model import XGBModel
    from xgboost import XGBClassifier
    XGBClassifierModel = XGBModel.class_from_sklearn_model_class(XGBClassifier)
    ```
    """

    XGB_MODEL_FILE = "model.txt"

    def _save(self, folder: Path) -> None:
        assert isinstance(self.model, BaseXGBModel)
        self.model.save_model(folder / self.XGB_MODEL_FILE)

    @classmethod
    def _load(cls: Type["XGBModel"], folder: Path, inputs: List[str], outputs: List[str]) -> "SklearnModel":
        model: T = cls._get_annotated_class()()
        model.load_model(folder / cls.XGB_MODEL_FILE)
        return cls(inputs=inputs, outputs=outputs, model=model, seed=1)
