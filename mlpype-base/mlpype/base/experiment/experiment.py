import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from mlpype.base.constants import Constants
from mlpype.base.data import DataCatalog, DataSet
from mlpype.base.evaluate import BaseEvaluator
from mlpype.base.experiment.argument_parsing import add_args_to_parser_for_pipeline
from mlpype.base.logger import ExperimentLogger
from mlpype.base.model import Model
from mlpype.base.pipeline import Pipeline
from mlpype.base.pipeline.type_checker import TypeChecker, TypeCheckerPipe
from mlpype.base.serialiser import JoblibSerialiser, Serialiser
from mlpype.base.utils.parsing import get_args_for_prefix


class Experiment:
    def __init__(
        self,
        data_sources: Dict[str, Union[DataCatalog, DataSet]],
        model: Model,
        pipeline: Pipeline,
        evaluator: BaseEvaluator,
        logger: ExperimentLogger,
        type_checker_classes: Optional[List[Type[TypeChecker]]] = None,
        input_type_checker: Optional[TypeCheckerPipe] = None,
        output_type_checker: Optional[TypeCheckerPipe] = None,
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
            data_sources (Dict[str, Dict[str, Union[DataCatalog, DataSet]]]): The
                DataCatalog or DataSets to use, in DataSource form. Should contain at least a 'train' DataSet.
                DataCatalogs will be initialised (`read`) in the beginning of the run. It is recommended to use
                DataCatalog in distributed cases, but for quick experimentation DataSets tend to be more useful.
                Please note that the DataSet will be changed by the Pipeline and Model. If you want to use a
                DataSet, you should make sure you don't modify the original data by reference or by key.
            model (Model): The Model to fit.
            pipeline (Pipeline): The Pipeline to use to transform data before feeding it to the Model.
            evaluator (BaseEvaluator): The evaluator to test how good your Model performs.
            logger (ExperimentLogger): The experiment logger to make sure you record how well your experiment worked,
                and log any artifacts such as the trained model.
            serialiser (Optional[Serialiser]): The serialiser to serialise any Python objects (expect the Model).
                Defaults to a joblib serialiser.
            output_folder: (Union[Path, str]): The output folder to log artifacts to. Defaults to "outputs".
            type_checker_classes (Optional[List[Type[TypeChecker]]]): A list of type checkers. Will be used to
                instantiate TypeCheckerPipe's for input and output datasets.
            input_type_checker: (Optional[TypeCheckerPipe]): A type checker for all input data. Will be used to verify
                incoming data and standardise the order of data. Will be used later to help serialise/deserialise data.
                Only select datasets required to do a run during inference.
                Not required if `type_checker_classes` is set.
            output_type_checker: (Optional[TypeCheckerPipe]): A type checker for all output data. Will be used to verify
                outgoing data and standardise the order of data. Will be used later to help serialise/deserialise data.
                Not required if `type_checker_classes` is set.
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

        if type_checker_classes is not None:
            # create the type checkers manually
            input_names = pipeline.get_input_datasets_names()
            output_names = model.outputs
            self.input_type_checker = TypeCheckerPipe(
                "_input_type_checker_", input_names, type_checker_classes=type_checker_classes
            )
            self.output_type_checker = TypeCheckerPipe(
                "_input_type_checker_", output_names, type_checker_classes=type_checker_classes
            )
        elif input_type_checker is not None and output_type_checker is not None:
            # use the pre-provided type checkers.
            self.input_type_checker = input_type_checker
            self.output_type_checker = output_type_checker
        else:
            raise ValueError(
                "Either type_checker_classes needs to be set or "
                "input_type_checker AND output_type_checker have to be set."
            )

        self.data_sources = data_sources
        self.model = model
        self.pipeline = pipeline
        self.evaluator = evaluator
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
            self.logger.info("Log parameters")
            self.experiment_logger.log_parameters(self.parameters)

            self.logger.info("Load data")
            datasets = {
                name: data_source_set.read() if isinstance(data_source_set, DataCatalog) else data_source_set
                for name, data_source_set in self.data_sources.items()
            }

            self.logger.info("Create input type checker")
            self.input_type_checker.fit(datasets["train"])
            input_checked = {name: self.input_type_checker.transform(ds) for name, ds in datasets.items()}

            self.logger.info("Fit pipeline")
            self.pipeline.fit(input_checked["train"])

            self.logger.info("Transform data")
            transformed = {
                name: self.pipeline.transform(data, is_inference=False) for name, data in input_checked.items()
            }

            self.logger.info("Fit model")
            self.model.fit(transformed["train"])

            self.logger.info("Evaluate model")
            metrics = {name: self.evaluator.evaluate(self.model, data) for name, data in transformed.items()}

            self.logger.info("Create output type checker")
            predicted_train = self.model.transform(transformed["train"])
            self.output_type_checker.fit(predicted_train)

            self.logger.info("Log results: metrics")
            for dataset_name, metric_set in metrics.items():
                self.experiment_logger.log_metrics(dataset_name, metric_set)

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
        """Logs the extra files for an experiment, as specified in the constructor.

        This also supports saving files outside of the current working directory.
        These files are saved under the root name of the file / folder you select.
        """
        paths_to_log = []
        self.logger.info("Log extra files")
        for extra_file in self.additional_files_to_store:
            try:
                # If the file to log is in the current work directory, use that to find
                # the relative path to the CWD. This is to help imports.
                source_path = Path(extra_file).relative_to(os.getcwd())
                logged_path = str(source_path)
            except ValueError:
                # The file is not on our current working directory. It is assumed you
                # added the file / folder to the system path in an alternative way.
                # It will be added as such.
                source_path = Path(extra_file).absolute()
                logged_path = Path(extra_file).name
            target_path = self.output_folder / logged_path
            self.experiment_logger.log_local_file(source_path, target_path)
            paths_to_log.append(logged_path)

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

        Old parameters will be kept, but overwritten if new values are provided.
        This makes it easier to make templates for hyperparameter tuning.

        Args:
            parameters (Dict[str, Any]): New parameters for the Model and Pipeline
                to be re-initialised with.
            seed (int, optional): Training seed. Defaults to 1.

        Returns:
            Experiment: A copy of this experiment, intialised with the new set
                of parameters.
        """
        new_params = self.parameters.copy()
        new_params.update(parameters)
        model_class = self.model.__class__

        model_args = get_args_for_prefix("model__", new_params)
        model = model_class(seed=seed, inputs=self.model.inputs, outputs=self.model.outputs, **model_args)

        pipeline_args = get_args_for_prefix("pipeline__", new_params)
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
            parameters=new_params,
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
        model_inputs: List[str],
        model_outputs: List[str],
        parameters: Dict[str, Any],
        type_checker_classes: Optional[List[Type[TypeChecker]]] = None,
        input_type_checker: Optional[TypeCheckerPipe] = None,
        output_type_checker: Optional[TypeCheckerPipe] = None,
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
            model_inputs: (List[str]): Input dataset names to the model.
            model_outputs: (List[str]): Output dataset names to the model.
            parameters (Optional[Dict[str, Any]]): Any parameters to log as part of this experiment.
                Defaults to None.
            type_checker_classes (Optional[List[Type[TypeChecker]]]): A list of type checkers. Will be used to
                instantiate TypeCheckerPipe's for input and output datasets.
            input_type_checker: (Optional[TypeCheckerPipe]): A type checker for all input data. Will be used to verify
                incoming data and standardise the order of data. Will be used later to help serialise/deserialise data.
                Only select datasets required to do a run during inference.
                Not required if `type_checker_classes` is set.
            output_type_checker: (Optional[TypeCheckerPipe]): A type checker for all output data. Will be used to verify
                outgoing data and standardise the order of data. Will be used later to help serialise/deserialise data.
                Not required if `type_checker_classes` is set.
            seed (int): The RNG seed to ensure reproducability.
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
            type_checker_classes=type_checker_classes,
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
        model_inputs: List[str],
        model_outputs: List[str],
        type_checker_classes: Optional[List[Type[TypeChecker]]] = None,
        input_type_checker: Optional[TypeCheckerPipe] = None,
        output_type_checker: Optional[TypeCheckerPipe] = None,
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
            model_inputs: (List[str]): Input dataset names to the model.
            model_outputs: (List[str]): Output dataset names to the model.
            type_checker_classes (Optional[List[Type[TypeChecker]]]): A list of type checkers. Will be used to
                instantiate TypeCheckerPipe's for input and output datasets.
            input_type_checker: (Optional[TypeCheckerPipe]): A type checker for all input data. Will be used to verify
                incoming data and standardise the order of data. Will be used later to help serialise/deserialise data.
                Only select datasets required to do a run during inference.
                Not required if `type_checker_classes` is set.
            output_type_checker: (Optional[TypeCheckerPipe]): A type checker for all output data. Will be used to verify
                outgoing data and standardise the order of data. Will be used later to help serialise/deserialise data.
                Not required if `type_checker_classes` is set.
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
            type_checker_classes=type_checker_classes,
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
