import json
from pathlib import Path
from typing import Type

from pype.base.constants import Constants
from pype.base.data import DataSet, DataSetSource
from pype.base.model import Model
from pype.base.pipeline import Pipeline
from pype.base.pipeline.type_checker import TypeCheckerPipe
from pype.base.serialiser.joblib_serialiser import JoblibSerialiser
from pype.base.utils.switch_workspace import switch_workspace


class Inferencer:
    def __init__(
        self,
        model: Model,
        pipeline: Pipeline,
        input_type_checker: TypeCheckerPipe,
        output_type_checker: TypeCheckerPipe,
    ):
        """Provides a standard way of inferencing with pype models.

        Args:
            model (Model): The Model to use in inference.
            pipeline (Pipeline): The Pipeline to use in inference.
            input_type_checker (TypeCheckerPipe): The type checker used to determine incoming types.
            output_type_checker (TypeCheckerPipe): The type checker used to determine outgoing types.
        """
        self.model = model
        self.pipeline = pipeline
        self.input_type_checker = input_type_checker
        self.output_type_checker = output_type_checker

    def predict(self, data: DataSet | DataSetSource) -> DataSet:
        """Predicts using the given data using the Pipeline and Model.

        # TODO: inverse transformation.

        Args:
            data (DataSet | DataSetSource): The data to predict for.

        Returns:
            DataSet: The predictions from Model.
        """
        if isinstance(data, DataSetSource):
            data = data.read()
        self.input_type_checker.transform(data)
        transformed = self.pipeline.transform(data)
        predicted = self.model.transform(transformed)
        self.output_type_checker.transform(predicted)
        return predicted

    @classmethod
    def from_folder(cls: Type["Inferencer"], folder: Path) -> "Inferencer":
        """Loads a Inferencer from the results of an Experiment.

        Args:
            folder (Path): The output folder from an Experiment, from which we load
                all required elements to make a inference pipeline.

        Returns:
            Inferencer: The inference pipeline that can predict for new data.
        """
        with open(folder / Constants.EXTRA_FILES, "r") as f:
            extra_files = json.load(f)["paths"]

        with switch_workspace(folder, extra_files):
            serialiser = JoblibSerialiser()
            model = Model.load(Constants.MODEL_FOLDER)
            pipeline = serialiser.deserialise(Constants.PIPELINE_FILE)
            input_type_checker = serialiser.deserialise(Constants.INPUT_TYPE_CHECKER_FILE)
            output_type_checker = serialiser.deserialise(Constants.OUTPUT_TYPE_CHECKER_FILE)

        return cls(
            model=model,
            pipeline=pipeline,
            input_type_checker=input_type_checker,
            output_type_checker=output_type_checker,
        )
