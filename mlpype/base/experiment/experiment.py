import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from mlpype.base.constants import Constants
from mlpype.base.data.data_catalog import DataCatalog
from mlpype.base.evaluate import BaseEvaluator
from mlpype.base.experiment.argument_parsing import add_args_to_parser_for_pipeline
from mlpype.base.logger import ExperimentLogger
from mlpype.base.model import Model
from mlpype.base.pipeline import Pipeline
from mlpype.base.pipeline.type_checker import TypeCheckerPipe
from mlpype.base.serialiser import JoblibSerialiser, Serialiser
from mlpype.base.utils.parsing import get_args_for_prefix


class Experiment:
    def __init__(
        self,
        data_sources: Dict[str, DataCatalog],
        model: Model,
        pipeline: Pipeline,
        evaluator: BaseEvaluator,
        logger: ExperimentLogger,
        input_type_checker: TypeCheckerPipe,
        output_type_checker: TypeCheckerPipe,
        serialiser: Optional[Serialiser] = None,
        output_folder: Union[Path, str] = "outputs",
        additional_files_to_store: Optional[List[Union[str, Path]]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """The core of the mlpype library: run a standardised ML experiment with the given parameters.

        It is highly recommended to use one of the class methods to initialise this object:
            - from_dictionary: If you've already got your parameters in the right form:
                - `model__<arg>` for model parameters
                - `pipeline__<pipe_name>__<arg> for pipeline parameters.

        Args:
            data_sources (Dict[str, DataCatalog]): The DataSets to use, in DataSource form.
                Should contain at least a 'train' DataSet. These will be initialised in the beginning of the run.
            model (Model): The Model to fit.
            pipeline (Pipeline): The Pipeline to use to transform data before feeding it to the Model.
            evaluator (BaseEvaluator): The evaluator to test how good your Model performs.
            logger (ExperimentLogger): The experiment logger to make sure you record how well your experiment worked,
                and log any artifacts such as the trained model.
            serialiser (Optional[Serialiser]): The serialiser to serialise any Python objects (expect the Model).
                Defaults to a joblib serialiser.
            output_folder: (Union[Path, str]): The output folder to log artifacts to. Defaults to "outputs".
            input_type_checker: (TypeCheckerPipe): A type checker for all input data. Will be used to verify incoming
                data and standardise the order of data. Will be used later to help serialise/deserialise data. Only
                select datasets required to do a run during inference.
            output_type_checker: (TypeCheckerPipe): A type checker for all output data. Will be used to verify outgoing
                data and standardise the order of data. Will be used later to help serialise/deserialise data.
            additional_files_to_store (Optional[List[Union[str, Path]]]): Extra files to store, such as python files.
                It is possible to select a directory as well, not just individual files.
                Defaults to no extra files (None).
            parameters (Optional[Dict[str, Any]]): Any parameters to log as part of this experiment.
                Defaults to None.
        """
        assert "train" in data_sources, "Must provide a 'train' entry in the data_sources dictionary."
        self.logger = getLogger(__name__)
        if serialiser is None:
            serialiser = JoblibSerialiser()

        if additional_files_to_store is None:
            additional_files_to_store = []
        if parameters is None:
            parameters = {}
            self.logger.warning(
                """It is highly recommended to provide the parameters used to initialise your
run here for logging purposes. Consider using the `from_command_line` or
`from_dictionary` initialisation methods"""
            )
        if isinstance(output_folder, str):
            output_folder = Path(output_folder)

        self.data_sources = data_sources
        self.model = model
        self.pipeline = pipeline
        self.evaluator = evaluator
        self.input_type_checker = input_type_checker
        self.output_type_checker = output_type_checker

        self.experiment_logger = logger
        self.parameters = parameters
        self.serialiser = serialiser
        self.output_folder = output_folder
        self.additional_files_to_store = additional_files_to_store

    def run(self) -> Dict[str, Dict[str, Union[str, float, int, bool]]]:
        """Execute the experiment.

        Returns:
            Dict[str, Union[str, float, int, bool]]: The performance metrics of this run.
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
            transformed = {name: self.pipeline.transform(data, is_inference=False) for name, data in datasets.items()}

            self.logger.info("Fit model")
            self.model.fit(transformed["train"])

            self.logger.info("Evaluate model")
            metrics = {name: self.evaluator.evaluate(self.model, data) for name, data in transformed.items()}

            self.logger.info("Create output type checker")
            predicted_train = self.model.transform(transformed["train"])
            self.output_type_checker.fit(predicted_train)

            self.logger.info("Log results: metrics, parameters")
            for dataset_name, metric_set in metrics.items():
                self.experiment_logger.log_metrics(dataset_name, metric_set)
            self.experiment_logger.log_parameters(self.parameters)

            self.logger.info("Log results: pipeline, model, serialiser")
            self._log_run()
        return metrics

    def _log_run(self) -> None:
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

        # log requirements.txt
        self._log_requirements()

        # extra py files
        self._log_extra_files()
        self.logger.info("Done")

        # log serialiser using a JoblibSerialiser
        jl_serialiser = JoblibSerialiser()
        self.experiment_logger.log_artifact(of / Constants.SERIALISER_FILE, jl_serialiser, object=self.serialiser)

    def _log_requirements(self) -> None:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        version_info = {
            "python_version": python_version,
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
        }
        version_file = self.output_folder / Constants.PYTHON_VERSION_FILE
        with open(version_file, "w") as f:
            json.dump(version_info, f)
        self.experiment_logger.log_file(version_file)

        reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
        requirements_file = self.output_folder / Constants.REQUIREMENTS_FILE
        with open(requirements_file, "w") as f:
            f.write(reqs)
        self.experiment_logger.log_file(requirements_file)

    def _log_extra_files(self) -> None:
        """Logs the extra files for an experiment, as specified in the constructor."""
        paths_to_log = []
        self.logger.info("Log extra files")
        for extra_file in self.additional_files_to_store:
            relative_path = Path(extra_file).relative_to(os.getcwd())
            self.experiment_logger.log_local_file(relative_path, self.output_folder / relative_path)
            paths_to_log.append(str(relative_path))

        self.logger.info("Log `extra files`-file")
        extra_files_file = self.output_folder / Constants.EXTRA_FILES
        with open(extra_files_file, "w") as f:
            data = {"paths": paths_to_log}
            json.dump(data, f)
        # Make sure the file path is closed before logging anything, to make sure all writes have flushed.
        self.experiment_logger.log_file(extra_files_file)

    def _create_output_folders(self) -> None:
        of = self.output_folder
        of.mkdir(exist_ok=True, parents=True)
        (of / Constants.MODEL_FOLDER).mkdir(exist_ok=True)

    def copy(self, parameters: Dict[str, Any], seed: int = 1) -> "Experiment":
        """Create a fresh copy of this Experiment.

        The Model & Pipeline will be recreated, so any trained versions will be
        re-initialised.

        Args:
            parameters (Dict[str, Any]): New parameters for the Model and Pipeline
                to be re-initialised with.
            seed (int, optional): Training seed. Defaults to 1.

        Returns:
            Experiment: _description_
        """
        model_class = self.model.__class__

        model_args = get_args_for_prefix("model__", parameters)
        model = model_class(seed=seed, inputs=self.model.inputs, outputs=self.model.outputs, **model_args)

        pipeline_args = get_args_for_prefix("pipeline__", parameters)
        new_pipeline = self.pipeline.copy(pipeline_args)

        return Experiment(
            # We might want to also deep copy data_sources, evaluator, type checkers, and experiment_logger.
            data_sources=self.data_sources,
            model=model,
            pipeline=new_pipeline,
            evaluator=self.evaluator,
            logger=self.experiment_logger,
            serialiser=self.serialiser,
            output_folder=self.output_folder,
            input_type_checker=self.input_type_checker,
            output_type_checker=self.output_type_checker,
            additional_files_to_store=self.additional_files_to_store,
            parameters=parameters,
        )

    @classmethod
    def from_dictionary(
        cls,
        data_sources: Dict[str, DataCatalog],
        model_class: Type[Model],
        pipeline: Pipeline,
        evaluator: BaseEvaluator,
        logger: ExperimentLogger,
        serialiser: Serialiser,
        input_type_checker: TypeCheckerPipe,
        output_type_checker: TypeCheckerPipe,
        model_inputs: List[str],
        model_outputs: List[str],
        parameters: Dict[str, Any],
        output_folder: Union[Path, str] = "outputs",
        seed: int = 1,
        additional_files_to_store: Optional[List[Union[str, Path]]] = None,
    ) -> "Experiment":
        """Creates an Experiment from a dictionary with parameters.

        Args:
            data_sources (Dict[str, DataCatalog]): The DataSets to use, in DataSource form.
                Should contain at least a 'train' DataSet. These will be initialised in the beginning of the run.
            model_class (Type[Model]): The class of the Model to fit.
            pipeline (Pipeline): The Pipeline to use to transform data before feeding it to the Model.
            evaluator (BaseEvaluator): The evaluator to test how good your Model performs.
            logger (ExperimentLogger): The experiment logger to make sure you record how well your experiment worked,
                and log any artifacts such as the trained model.
            serialiser (Serialiser): The serialiser to serialise any Python objects (expect the Model).
            output_folder: (Union[Path, str]): The output folder to log artifacts to.
            input_type_checker: (TypeCheckerPipe): A type checker for all input data. Will be used to verify incoming
                data and standardise the order of data. Will be used later to help serialise/deserialise data.
            output_type_checker: (TypeCheckerPipe): A type checker for all output data. Will be used to verify outgoing
                data and standardise the order of data. Will be used later to help serialise/deserialise data.
            model_inputs: (List[str]): Input dataset names to the model.
            model_outputs: (List[str]): Output dataset names to the model.
            seed (int): The RNG seed to ensure reproducability.
            parameters (Optional[Dict[str, Any]]): Any parameters to log as part of this experiment.
                Defaults to None.
            additional_files_to_store (Optional[List[Union[str, Path]]]): Extra files to store, such as python files.
                Defaults to no extra files (None).

        Returns:
            Experiment: An Experiment created with the given parameters.
        """
        model_args = get_args_for_prefix("model__", parameters)
        model = model_class(seed=seed, inputs=model_inputs, outputs=model_outputs, **model_args)

        pipeline_args = get_args_for_prefix("pipeline__", parameters)
        pipeline.reinitialise(pipeline_args)

        return Experiment(
            data_sources=data_sources,
            model=model,
            pipeline=pipeline,
            evaluator=evaluator,
            logger=logger,
            serialiser=serialiser,
            output_folder=output_folder,
            input_type_checker=input_type_checker,
            output_type_checker=output_type_checker,
            additional_files_to_store=additional_files_to_store,
            parameters=parameters,
        )

    @classmethod
    def from_command_line(
        cls,
        data_sources: Dict[str, DataCatalog],
        model_class: Type[Model],
        pipeline: Pipeline,
        evaluator: BaseEvaluator,
        logger: ExperimentLogger,
        serialiser: Serialiser,
        input_type_checker: TypeCheckerPipe,
        output_type_checker: TypeCheckerPipe,
        model_inputs: List[str],
        model_outputs: List[str],
        output_folder: Union[Path, str] = "outputs",
        seed: int = 1,
        additional_files_to_store: Optional[List[Union[str, Path]]] = None,
        fixed_arguments: Optional[Dict[str, Any]] = None,
    ) -> "Experiment":
        """Automatically initialises an Experiment from command line arguments.

        Args:
            data_sources (Dict[str, DataCatalog]): The DataSets to use, in DataSource form.
                Should contain at least a 'train' DataSet. These will be initialised in the beginning of the run.
            model_class (Type[Model]): The class of the Model to fit.
            pipeline (Pipeline): The Pipeline to use to transform data before feeding it to the Model.
            evaluator (BaseEvaluator): The evaluator to test how good your Model performs.
            logger (ExperimentLogger): The experiment logger to make sure you record how well your experiment worked,
                and log any artifacts such as the trained model.
            serialiser (Serialiser): The serialiser to serialise any Python objects (expect the Model).
            output_folder: (Union[Path, str]): The output folder to log artifacts to.
            input_type_checker: (TypeCheckerPipe): A type checker for all input data. Will be used to verify incoming
                data and standardise the order of data. Will be used later to help serialise/deserialise data.
            output_type_checker: (TypeCheckerPipe): A type checker for all output data. Will be used to verify outgoing
                data and standardise the order of data. Will be used later to help serialise/deserialise data.
            model_inputs: (List[str]): Input dataset names to the model.
            model_outputs: (List[str]): Output dataset names to the model.
            seed (int): The RNG seed to ensure reproducability.
            additional_files_to_store (Optional[List[Union[str, Path]]], optional): Extra files to store,
                such as python files. Defaults to no extra files (None).
            fixed_arguments (Optional[Dict[str, Any]]): Arguments that won't be read from command line.
                Useful to pass complex objects that you don't want to optimize. Think of classes, loss functions,
                metrics, etc. Any value in this dict will overwrite values from the command line. These need to be
                presented in the same `model__<arg>` or `pipeline__<step>__<arg>` format as the command line arguments.


        Returns:
            Experiment: An Experiment created with the given parameters from the command line.
        """
        arg_parser = cls._get_cmd_args(model_class, pipeline)
        parsed_args, _ = arg_parser.parse_known_args()

        arguments = parsed_args.__dict__
        if fixed_arguments is not None:
            arguments.update(fixed_arguments)

        return cls.from_dictionary(
            data_sources=data_sources,
            model_class=model_class,
            pipeline=pipeline,
            evaluator=evaluator,
            logger=logger,
            serialiser=serialiser,
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

    def __str__(self) -> str:
        """Create string representation of this Experiment.

        Returns:
            str: A string representation of this Experiment.
        """
        data_part = "datasets:\n" + "\n\t".join(
            [
                f"""\t{name}:
{ds_source.__str__(indents=2)}"""
                for name, ds_source in self.data_sources.items()
            ]
        )

        pipeline_part = "pipeline:\n" + self.pipeline.__str__(indents=1)
        model_part = f"model:\n\t{str(self.model)}"
        logger_part = f"logger:\n\t{str(self.experiment_logger)}"
        evaluator_part = f"evaluator:\n{self.evaluator.__str__(indents=1)}"

        return "\n\n".join([data_part, pipeline_part, model_part, logger_part, evaluator_part])
