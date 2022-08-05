import warnings
from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path
from typing import Any, Type

from pype.base.constants import Constants
from pype.base.data.dataset_source import DataSetSource
from pype.base.evaluate import Evaluator
from pype.base.experiment.argument_parsing import add_args_to_parser_for_pipeline
from pype.base.logger import ExperimentLogger
from pype.base.model import Model
from pype.base.pipeline import Pipeline
from pype.base.pipeline.type_checker import TypeCheckerPipe
from pype.base.serialiser.serialiser import Serialiser
from pype.base.utils.parsing import get_args_for_prefix


class Experiment:
    def __init__(
        self,
        data_sources: dict[str, DataSetSource],
        model: Model,
        pipeline: Pipeline,
        evaluator: Evaluator,
        logger: ExperimentLogger,
        serialiser: Serialiser,
        output_folder: Path | str,
        input_type_checker: TypeCheckerPipe,
        output_type_checker: TypeCheckerPipe,
        additional_files_to_store: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
    ):
        """The core of the pype library: run a standardised ML experiment with the given parameters.

        It is highly recommended to use one of the class methods to initialise this object:
            - from_dictionary: If you've already got your parameters in the right form:
                - `model__<arg>` for model parameters
                - `pipeline__<pipe_name>__<arg> for pipeline parameters.

        Args:
            data_sources (dict[str, DataSetSource]): The DataSets to use, in DataSource form.
                Should contain at least a 'train' DataSet. These will be initialised in the beginning of the run.
            model (Model): The Model to fit.
            pipeline (Pipeline): The Pipeline to use to transform data before feeding it to the Model.
            evaluator (Evaluator): The evaluator to test how good your Model performs.
            logger (ExperimentLogger): The experiment logger to make sure you record how well your experiment worked,
                and log any artifacts such as the trained model.
            serialiser (Serialiser): The serialiser to serialise any Python objects (expect the Model).
            output_folder: (Path | str): The output folder to log artifacts to.
            input_type_checker: (TypeCheckerPipe): A type checker for all input data. Will be used to verify incoming
                data and standardise the order of data. Will be used later to help serialise/deserialise data.
            output_type_checker: (TypeCheckerPipe): A type checker for all output data. Will be used to verify outgoing
                data and standardise the order of data. Will be used later to help serialise/deserialise data.
            additional_files_to_store (list[str] | None, optional): Extra files to store, such as python files.
                Defaults to no extra files (None).
            parameters (dict[str, Any] | None, optional): Any parameters to log as part of this experiment.
                Defaults to None.
        """
        assert "train" in data_sources, "Must provide a 'train' entry in the data_sources dictionary."
        if additional_files_to_store is None:
            additional_files_to_store = []
        if parameters is None:
            parameters = {}
            warnings.warn(
                """
It is highly recommended to provide the parameters used to initialise your
run here for logging purposes. Consider using the `from_command_line` or
`from_dictionary` initialisation methods
                """
            )
        if isinstance(output_folder, str):
            output_folder = Path(output_folder)

        self.data_sources = data_sources
        self.model = model
        self.pipeline = pipeline
        self.evaluator = evaluator
        self.input_type_checker = input_type_checker
        self.output_type_checker = output_type_checker

        self.logger = getLogger(__name__)
        self.experiment_logger = logger
        self.parameters = parameters
        self.serialiser = serialiser
        self.output_folder = output_folder
        self.additional_files_to_store = additional_files_to_store

    def run(self) -> dict[str, dict[str, str | float | int | bool]]:
        """Execute the experiment.

        Returns:
            dict[str, dict[str, str | float | int | bool]]: The performance metrics of this run.
        """
        with self.experiment_logger:
            self.logger.info("Load data")
            datasets = {name: data_source_set.read() for name, data_source_set in self.data_sources.items()}

            self.logger.info("Create input type checker")
            self.input_type_checker.fit(datasets["train"])
            for ds in datasets.values():
                self.input_type_checker.transform(ds)

            self.logger.info("Fit pipeline")
            self.pipeline.fit(datasets["train"])

            self.logger.info("Transform data")
            transformed = {name: self.pipeline.transform(data) for name, data in datasets.items()}

            self.logger.info("Fit model")
            self.model.fit(transformed["train"])

            self.logger.info("Evaluate model")
            metrics = {name: self.evaluator.evaluate(self.model, data) for name, data in datasets.items()}

            self.logger.info("Create output type checker")
            predicted_train = self.model.transform(transformed["train"])
            self.output_type_checker.fit(predicted_train)

            self.logger.info("Log results: metrics, parameters, pipeline, model")
            for dataset_name, metric_set in metrics.items():
                self.experiment_logger.log_metrics(dataset_name, metric_set)

            self._create_output_folders()

            of = self.output_folder
            self.experiment_logger.log_model(self.model, of / Constants.MODEL_FOLDER)
            self.experiment_logger.log_artifact(of / Constants.PIPELINE_FILE, self.serialiser, object=self.pipeline)
            self.experiment_logger.log_artifact(
                of / Constants.INPUT_TYPE_CHECKER_FILE, self.serialiser, object=self.input_type_checker
            )
            self.experiment_logger.log_artifact(
                of / Constants.OUTPUT_TYPE_CHECKER_FILE, self.serialiser, object=self.output_type_checker
            )
            self.experiment_logger.log_parameters(self.parameters)

            for extra_file in self.additional_files_to_store:
                self.experiment_logger.log_file(extra_file)

            self.logger.info("Done")
        return metrics

    def _create_output_folders(self) -> None:
        of = self.output_folder
        of.mkdir(exist_ok=True, parents=True)
        (of / Constants.MODEL_FOLDER).mkdir(exist_ok=True)

    @classmethod
    def from_dictionary(
        cls,
        data_sources: dict[str, DataSetSource],
        model_class: Type[Model],
        pipeline: Pipeline,
        evaluator: Evaluator,
        logger: ExperimentLogger,
        serialiser: Serialiser,
        output_folder: Path | str,
        input_type_checker: TypeCheckerPipe,
        output_type_checker: TypeCheckerPipe,
        model_inputs: list[str],
        model_outputs: list[str],
        parameters: dict[str, Any],
        seed: int = 1,
        additional_files_to_store: list[str] | None = None,
    ) -> "Experiment":
        """Creates an Experiment from a dictionary with parameters.

        Args:
            data_sources (dict[str, DataSetSource]): The DataSets to use, in DataSource form.
                Should contain at least a 'train' DataSet. These will be initialised in the beginning of the run.
            model_class (Type[Model]): The class of the Model to fit.
            pipeline (Pipeline): The Pipeline to use to transform data before feeding it to the Model.
            evaluator (Evaluator): The evaluator to test how good your Model performs.
            logger (ExperimentLogger): The experiment logger to make sure you record how well your experiment worked,
                and log any artifacts such as the trained model.
            serialiser (Serialiser): The serialiser to serialise any Python objects (expect the Model).
            output_folder: (Path | str): The output folder to log artifacts to.
            input_type_checker: (TypeCheckerPipe): A type checker for all input data. Will be used to verify incoming
                data and standardise the order of data. Will be used later to help serialise/deserialise data.
            output_type_checker: (TypeCheckerPipe): A type checker for all output data. Will be used to verify outgoing
                data and standardise the order of data. Will be used later to help serialise/deserialise data.
            model_inputs: (list[str]): Input dataset names to the model.
            model_outputs: (list[str]): Output dataset names to the model.
            seed (int): The RNG seed to ensure reproducability.
            parameters (dict[str, Any] | None, optional): Any parameters to log as part of this experiment.
                Defaults to None.
            additional_files_to_store (list[str] | None, optional): Extra files to store, such as python files.
                Defaults to no extra files (None).

        Returns:
            Experiment: An Experiment created with the given parameters.
        """
        model_args = get_args_for_prefix("model__", parameters)
        model = model_class(seed=seed, inputs=model_inputs, outputs=model_outputs, **model_args)

        pipeline_args = get_args_for_prefix("pipeline__", parameters)
        pipeline.reinitialise(pipeline_args)

        return Experiment(
            data_sources,
            model,
            pipeline,
            evaluator,
            logger,
            serialiser,
            output_folder,
            input_type_checker=input_type_checker,
            output_type_checker=output_type_checker,
            additional_files_to_store=additional_files_to_store,
            parameters=parameters,
        )

    @classmethod
    def from_command_line(
        cls,
        data_sources: dict[str, DataSetSource],
        model_class: Type[Model],
        pipeline: Pipeline,
        evaluator: Evaluator,
        logger: ExperimentLogger,
        serialiser: Serialiser,
        output_folder: Path | str,
        input_type_checker: TypeCheckerPipe,
        output_type_checker: TypeCheckerPipe,
        model_inputs: list[str],
        model_outputs: list[str],
        seed: int = 1,
        additional_files_to_store: list[str] | None = None,
    ) -> "Experiment":
        """Automatically initialises an Experiment from command line arguments.

        # TODO: how to best store these extra files to the output folder as well?

        Args:
            data_sources (dict[str, DataSetSource]): The DataSets to use, in DataSource form.
                Should contain at least a 'train' DataSet. These will be initialised in the beginning of the run.
            model_class (Type[Model]): The class of the Model to fit.
            pipeline (Pipeline): The Pipeline to use to transform data before feeding it to the Model.
            evaluator (Evaluator): The evaluator to test how good your Model performs.
            logger (ExperimentLogger): The experiment logger to make sure you record how well your experiment worked,
                and log any artifacts such as the trained model.
            serialiser (Serialiser): The serialiser to serialise any Python objects (expect the Model).
            output_folder: (Path | str): The output folder to log artifacts to.
            input_type_checker: (TypeCheckerPipe): A type checker for all input data. Will be used to verify incoming
                data and standardise the order of data. Will be used later to help serialise/deserialise data.
            output_type_checker: (TypeCheckerPipe): A type checker for all output data. Will be used to verify outgoing
                data and standardise the order of data. Will be used later to help serialise/deserialise data.
            model_inputs: (list[str]): Input dataset names to the model.
            model_outputs: (list[str]): Output dataset names to the model.
            seed (int): The RNG seed to ensure reproducability.
            additional_files_to_store (list[str] | None, optional): Extra files to store, such as python files.
                Defaults to no extra files (None).

        Returns:
            Experiment: An Experiment created with the given parameters from the command line.
        """
        arg_parser = cls._get_cmd_args(model_class, pipeline)
        parsed_args, _ = arg_parser.parse_known_args()

        return cls.from_dictionary(
            data_sources,
            model_class,
            pipeline,
            evaluator,
            logger,
            serialiser,
            output_folder=output_folder,
            input_type_checker=input_type_checker,
            output_type_checker=output_type_checker,
            additional_files_to_store=additional_files_to_store,
            parameters=parsed_args.__dict__,
            seed=seed,
            model_inputs=model_inputs,
            model_outputs=model_outputs,
        )

    @classmethod
    def _get_cmd_args(cls, model_class: Type[Model], pipeline: Pipeline) -> ArgumentParser:
        parser = ArgumentParser()

        model_class.get_parameters(parser)
        add_args_to_parser_for_pipeline(parser, pipeline)
        return parser
