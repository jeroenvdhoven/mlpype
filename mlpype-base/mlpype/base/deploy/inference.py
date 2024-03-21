import json
from pathlib import Path
from typing import Optional, Type, Union

from mlpype.base.constants import Constants
from mlpype.base.data import DataCatalog, DataSet
from mlpype.base.experiment.experiment import Experiment
from mlpype.base.model import Model
from mlpype.base.pipeline import Pipeline
from mlpype.base.pipeline.type_checker import TypeCheckerPipe
from mlpype.base.serialiser.joblib_serialiser import JoblibSerialiser
from mlpype.base.serialiser.serialiser import Serialiser
from mlpype.base.utils.workspace import switch_workspace


class Inferencer:
    def __init__(
        self,
        model: Model,
        pipeline: Pipeline,
        input_type_checker: TypeCheckerPipe,
        output_type_checker: TypeCheckerPipe,
    ):
        """Provides a standard way of inferencing with mlpype models.

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

    def predict(self, data: Union[DataSet, DataCatalog], return_transformed_data: bool = False) -> DataSet:
        """Predicts using the given data using the Pipeline and Model.

        Args:
            data (Union[DataSet, DataCatalog]): The data to predict for.
            return_transformed_data (bool): Flag indicating if transformed data
                also needs to be returned. If set to True, we still only return one
                dataset, but we'll add the data after the pipeline has run to it.

        Returns:
            DataSet: The predictions from Model, with transformed data from the Pipeline if
                requested.
        """
        # TODO: inverse transformation after prediction
        if isinstance(data, DataCatalog):
            data = data.read()
        checked_input = self.input_type_checker.transform(data)
        transformed = self.pipeline.transform(checked_input, is_inference=True)
        predicted = self.model.transform(transformed)
        checked_output = self.output_type_checker.transform(predicted)

        if return_transformed_data:
            transformed.set_all(checked_output.keys(), checked_output.values())
            return transformed
        return checked_output

    @classmethod
    def from_folder(cls: Type["Inferencer"], folder: Path, serialiser: Optional[Serialiser] = None) -> "Inferencer":
        """Loads a Inferencer from the results of an Experiment.

        We use the absolute version of the path to try and prevent loading issues.

        Args:
            folder (Path): The output folder from an Experiment, from which we load
                all required elements to make a inference pipeline.
            serialiser (Optional[Serialiser]): The Serialiser used to deserialise the pipeline and input/output
                type checkers. Defaults to a JoblibSerialiser.

        Returns:
            Inferencer: The inference pipeline that can predict for new data.
        """
        folder = folder.absolute()
        with open(folder / Constants.EXTRA_FILES, "r") as f:
            extra_files = json.load(f)["paths"]

        with switch_workspace(folder, extra_files):
            if serialiser is None:
                tmp_serialiser = JoblibSerialiser()
                serialiser = tmp_serialiser.deserialise(folder / Constants.SERIALISER_FILE)
            assert isinstance(serialiser, Serialiser), "Please provide a Serialiser!"

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
