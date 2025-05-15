"""Provides an integration layer between MLpype-trained models and MLflow models."""
from pathlib import Path

from loguru import logger
from mlflow.pyfunc import PythonModel, PythonModelContext

from mlpype.base.data.dataset import DataSet
from mlpype.base.deploy.inference import Inferencer


class PypeMLFlowModel(PythonModel):
    """An integration layer between MLpype-trained models and MLflow models."""

    def load_context(self, context: PythonModelContext) -> None:
        """
        Loads the context for the Python model.

        This load an Inferencer from the logged folder.

        Parameters:
            context (PythonModelContext): The default context provided by MLpype + MLflow.
                This included 1 variable, the path to the logged folder.
                This is the lower level artifact folder: MLPype stores
                the results higher up in the folder structure.
        """
        assert "folder" in context.artifacts
        root = Path(context.artifacts["folder"])
        print(f"Starting in {root}")
        for _ in range(4):
            if (root / "extra_files.json").exists():
                break
            root = root.parent
            print(f"Trying {root}")
        else:
            raise ValueError("Could not locate proper root folder to import model.")
        self.inferencer = Inferencer.from_folder(root)

    def predict(self, context: PythonModelContext, model_input: DataSet) -> DataSet:
        """
        Makes a predition using the loaded MLpype Inferencer.

        Args:
            context (PythonModelContext): The context for the Python model. Not used,
                but required as a default argument.
            model_input (DataSet): The input data for the model prediction.

        Returns:
            DataSet: The predicted output dataset.
        """
        return self.inferencer.predict(model_input)

    def __getstate__(self) -> dict:
        """Gets the state of this object, excluding any Inferencer."""
        state = self.__dict__.copy()
        if "inferencer" in state:
            del state["inferencer"]
        return state

    def __setstate__(self, state: dict) -> None:
        """Sets the state of this object, ignoring any Inferencer."""
        logger.warning(
            "PypeMLFlowModel is not intended to be loaded using pickle. "
            "Please use `load_context` or otherwise properly initialise this class"
        )
        state["inferencer"] = None
        self.__dict__ = state
