"""The core of the mlpype library: run a standardised ML experiment with the given parameters."""
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from loguru import logger as loguru_logger

from mlpype.base.data import DataCatalog, DataSet
from mlpype.base.evaluate import BaseEvaluator
from mlpype.base.evaluate.plot import BasePlotter
from mlpype.base.experiment.argument_parsing import add_args_to_parser_for_pipeline
from mlpype.base.logger import ExperimentLogger
from mlpype.base.model import Model
from mlpype.base.pipeline import Pipeline
from mlpype.base.pipeline.type_checker import TypeChecker, TypeCheckerPipe
from mlpype.base.serialiser import JoblibSerialiser, Serialiser
from mlpype.base.utils.parsing import get_args_for_prefix


class Experiment:
    """The core of the mlpype library: run a standardised ML experiment with the given parameters.

    This combines all the components of the mlpype library to run a standardised ML experiment.
    DataCatalogs will be initialised (`read`) in the beginning of the run. After this,
    the Pipeline will be fitted on the train data, and the Model will be trained on the
    transformed train data. After this, the Model will be evaluated on all data.

    All metrics, parameters, model and pipeline artifacts, and plots will be logged. Finally,
    the performance metrics will be returned.

    A trained/logged Experiment can be reloaded using the Inferencer class and its `from_folder` method.

    It is highly recommended to use one of the class methods to initialise this object:
            - from_dictionary: If you've already got your parameters in the right form:
                - `model__<arg>` for model parameters
                - `pipeline__<pipe_name>__<arg> for pipeline parameters.
    """

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
        plots: Optional[List[BasePlotter]] = None,
    ):
        """The core of the mlpype library: run a standardised ML experiment with the given parameters.

        It is highly recommended to use one of the class methods to initialise this object:
            - from_dictionary: If you've already got your parameters in the right form:
                - `model__<arg>` for model parameters
                - `pipeline__<pipe_name>__<arg> for pipeline parameters.

        Args:
            data_sources (Dict[str, Union[DataCatalog, DataSet]]): The DataCatalog or DataSets to use, in DataSource
                form. Should contain at least a 'train' DataSet.
                DataCatalogs will be initialised (`read`) in the beginning of the run. It is recommended to use
                DataCatalog in distributed cases, but for quick experimentation DataSets tend to be more useful.
                Please note that the DataSet will be changed by the Pipeline and Model. If you want to use a
                DataSet, you should make sure you don't modify the original data by reference or by key.
            model (Model): The Model to fit.
            pipeline (Pipeline): The Pipeline to use to transform data before feeding it to the Model.
            evaluator (BaseEvaluator): The evaluator to test how good your Model performs.
            logger (ExperimentLogger): The experiment logger to make sure you record how well your experiment worked,
                and log any artifacts such as the trained model.
            type_checker_classes (Optional[List[Type[TypeChecker]]]): A list of type checkers. Will be used to
                instantiate TypeCheckerPipe's for input and output datasets.
            input_type_checker (Optional[TypeCheckerPipe]): A type checker for all input data. Will be used to verify
                incoming data and standardise the order of data. Will be used later to help serialise/deserialise data.
                Not required if `type_checker_classes` is set.
            output_type_checker (Optional[TypeCheckerPipe]): A type checker for all output data. Will be used to verify
                outgoing data and standardise the order of data. Will be used later to help serialise/deserialise data.
                Not required if `type_checker_classes` is set.
            serialiser (Optional[Serialiser]): The serialiser to serialise any Python objects (expect the Model).
                Defaults to a joblib serialiser.
            output_folder (Union[Path, str]): The output folder to log artifacts to. Defaults to "outputs".
            additional_files_to_store (Optional[List[Union[str, Path]]]): Extra files to store, such as python files.
                It is possible to select a directory as well, not just individual files.
                Defaults to no extra files (None).
            parameters (Optional[Dict[str, Any]]): Any parameters to log as part of this experiment.
                Defaults to None.
            plots (Optional[List[BasePlotter]]): A list of plots to generate. Defaults to None, resulting in no plots.
                All plots will be written to the outputs folder.
        """
        assert "train" in data_sources, "Must provide a 'train' entry in the data_sources dictionary."
        if serialiser is None:
            serialiser = JoblibSerialiser()

        if plots is None:
            plots = []

        if additional_files_to_store is None:
            additional_files_to_store = []
        if parameters is None:
            parameters = {}
            loguru_logger.warning(
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
        self.plots = plots

    def run(self) -> Dict[str, Dict[str, Union[str, float, int, bool]]]:
        """Execute the experiment.

        Returns:
            Dict[str, Dict[str, Union[str, float, int, bool]]]: The performance metrics of this run.
        """
        with self.experiment_logger:
            loguru_logger.info("Log parameters")
            self.experiment_logger.log_parameters(self.parameters)

            loguru_logger.info("Load data")
            datasets = {
                name: data_source_set.read() if isinstance(data_source_set, DataCatalog) else data_source_set
                for name, data_source_set in self.data_sources.items()
            }

            loguru_logger.info("Create input type checker")
            self.input_type_checker.fit(datasets["train"])
            input_checked = {name: self.input_type_checker.transform(ds) for name, ds in datasets.items()}

            loguru_logger.info("Fit pipeline")
            self.pipeline.fit(input_checked["train"])

            loguru_logger.info("Transform data")
            transformed = {
                name: self.pipeline.transform(data, is_inference=False) for name, data in input_checked.items()
            }

            loguru_logger.info("Fit model")
            self.model.fit(transformed["train"])

            loguru_logger.info("Evaluate model")
            metrics = {name: self.evaluator.evaluate(self.model, data) for name, data in transformed.items()}

            loguru_logger.info("Create output type checker")
            predicted_train = self.model.transform(transformed["train"])
            self.output_type_checker.fit(predicted_train)

            loguru_logger.info("Log results.")
            self.experiment_logger.log_run(self, metrics, transformed)
            loguru_logger.info("Done")
        return metrics

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
        data_sources: Dict[str, Union[DataCatalog, DataSet]],
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
            data_sources (Dict[str, Union[DataCatalog, DataSet]]): The DataSets to use, in DataSource form.
                Should contain at least a 'train' DataSet. These will be initialised in the beginning of the run.
            model_class (Type[Model]): The class of the Model to fit.
            pipeline (Pipeline): The Pipeline to use to transform data before feeding it to the Model.
            evaluator (BaseEvaluator): The evaluator to test how good your Model performs.
            logger (ExperimentLogger): The experiment logger to make sure you record how well your experiment worked,
                and log any artifacts such as the trained model.
            serialiser (Serialiser): The serialiser to serialise any Python objects (expect the Model).
            model_inputs (List[str]): Input dataset names to the model.
            model_outputs (List[str]): Output dataset names to the model.
            parameters (Dict[str, Any]): Any parameters to log as part of this experiment.
                Defaults to None.
            type_checker_classes (Optional[List[Type[TypeChecker]]]): A list of type checkers. Will be used to
                instantiate TypeCheckerPipe's for input and output datasets.
            input_type_checker (Optional[TypeCheckerPipe]): A type checker for all input data. Will be used to verify
                incoming data and standardise the order of data. Will be used later to help serialise/deserialise data.
                Only select datasets required to do a run during inference.
                Not required if `type_checker_classes` is set.
            output_type_checker (Optional[TypeCheckerPipe]): A type checker for all output data. Will be used to verify
                outgoing data and standardise the order of data. Will be used later to help serialise/deserialise data.
                Not required if `type_checker_classes` is set.
            output_folder (Union[Path, str]): The output folder to log artifacts to.
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
        data_sources: Dict[str, Union[DataCatalog, DataSet]],
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
            data_sources (Dict[str, Union[DataCatalog, DataSet]]): The DataSets to use, in DataSource form.
                Should contain at least a 'train' DataSet. These will be initialised in the beginning of the run.
            model_class (Type[Model]): The class of the Model to fit.
            pipeline (Pipeline): The Pipeline to use to transform data before feeding it to the Model.
            evaluator (BaseEvaluator): The evaluator to test how good your Model performs.
            logger (ExperimentLogger): The experiment logger to make sure you record how well your experiment worked,
                and log any artifacts such as the trained model.
            serialiser (Serialiser): The serialiser to serialise any Python objects (expect the Model).
            model_inputs (List[str]): Input dataset names to the model.
            model_outputs (List[str]): Output dataset names to the model.
            type_checker_classes (Optional[List[Type[TypeChecker]]]): A list of type checkers. Will be used to
                instantiate TypeCheckerPipe's for input and output datasets.
            input_type_checker (Optional[TypeCheckerPipe]): A type checker for all input data. Will be used to verify
                incoming data and standardise the order of data. Will be used later to help serialise/deserialise data.
                Only select datasets required to do a run during inference.
                Not required if `type_checker_classes` is set.
            output_type_checker (Optional[TypeCheckerPipe]): A type checker for all output data. Will be used to verify
                outgoing data and standardise the order of data. Will be used later to help serialise/deserialise data.
                Not required if `type_checker_classes` is set.
            output_folder (Union[Path, str]): The output folder to log artifacts to.
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
