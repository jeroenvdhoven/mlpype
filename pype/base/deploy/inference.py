import json
from pathlib import Path
from typing import Type

from pype.base.constants import Constants
from pype.base.data import DataSet, DataSetSource
from pype.base.experiment.experiment import Experiment
from pype.base.model import Model
from pype.base.pipeline import Pipeline
from pype.base.pipeline.type_checker import TypeCheckerPipe
from pype.base.serialiser.joblib_serialiser import JoblibSerialiser
from pype.base.serialiser.serialiser import Serialiser
from pype.base.utils.workspace import switch_workspace


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
    def from_folder(cls: Type["Inferencer"], folder: Path, serialiser: Serialiser | None = None) -> "Inferencer":
        """Loads a Inferencer from the results of an Experiment.

        We use the absolute version of the path to try and prevent loading issues.

        Args:
            folder (Path): The output folder from an Experiment, from which we load
                all required elements to make a inference pipeline.
            serialiser (Serialiser | None): The Serialiser used to deserialise the pipeline and input/output
                type checkers. Defaults to a JoblibSerialiser.

        Returns:
            Inferencer: The inference pipeline that can predict for new data.
        """
        if serialiser is None:
            serialiser = JoblibSerialiser()

        folder = folder.absolute()
        with open(folder / Constants.EXTRA_FILES, "r") as f:
            extra_files = json.load(f)["paths"]

        with switch_workspace(folder, extra_files):
            model = Model.load(folder / Constants.MODEL_FOLDER)
            pipeline = serialiser.deserialise(folder / Constants.PIPELINE_FILE)
            input_type_checker = serialiser.deserialise(folder / Constants.INPUT_TYPE_CHECKER_FILE)
            output_type_checker = serialiser.deserialise(folder / Constants.OUTPUT_TYPE_CHECKER_FILE)

        return cls(
            model=model,
            pipeline=pipeline,
            input_type_checker=input_type_checker,
            output_type_checker=output_type_checker,
        )

    @classmethod
    def from_experiment(cls, experiment: Experiment) -> "Inferencer":
        """Generates an Inferencer from an experiment.

        We currently cannot check if the experiment has been trained already.

        Args:
            experiment (Experiment): The experiment to convert to a inferencing
                pipeline.

        Returns:
            Inferencer: An inferencing pipeline obtained from the given experiment.
        """
        return cls(
            model=experiment.model,
            pipeline=experiment.pipeline,
            input_type_checker=experiment.input_type_checker,
            output_type_checker=experiment.output_type_checker,
        )

    def __str__(self) -> str:
        """Create string representation of this Inferencer.

        Returns:
            str: A string representation of this Inferencer.
        """
        pipeline_part = "pipeline:\n" + self.pipeline.__str__(indents=1)
        model_part = f"model:\n\t{str(self.model)}"

        inputs = f"inputs:\n\t{str(self.input_type_checker)}"
        outputs = f"outputs:\n\t{str(self.output_type_checker)}"

        return "\n\n".join([inputs, outputs, pipeline_part, model_part])
