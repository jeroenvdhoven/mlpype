from typing import Any, Dict

from lightgbm import LGBMModel

from mlpype.sklearn.model.sklearn_model import SklearnModel


class LightGBMModel(SklearnModel[LGBMModel]):
    """A mlpype implementation of lightgbm using the Sklearn interface.

    For Mac users: please keep in mind that OpenMP may not be installed
    correctly for lightgbm to use it. If your training hangs, set
    the number of threads to 1.
    """

    def _init_model(self, args: Dict[str, Any]) -> LGBMModel:
        """Initialise a new lgb.LGBMModel model.

        Args:
            args (Dict[str, Any]): The arguments to initialise a LGBMModel.

        Returns:
            LGBMModel: A new LGBMModel.
        """
        return LGBMModel(**args)
