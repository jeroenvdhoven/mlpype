from pathlib import Path

from pype.base.data import DataSet, DataSetSource
from pype.base.deploy import Inferencer


class Predictor:
    def __init__(self) -> None:
        """A basic interface to the fixed Inferencer in this package."""
        folder = Path(__file__).parent / "outputs"
        print(f"Loading inferencer from {str(folder)}")
        self.inferencer = Inferencer.from_folder(folder)

    def predict(self, data: DataSet | DataSetSource) -> DataSet:
        """Uses the initialised Inferencer to make a prediction.

        Args:
            data (DataSet | DataSetSource): Input data, see `Inferencer`

        Returns:
            DataSet: Output data, see `Inferencer`
        """
        return self.inferencer.predict(data)
