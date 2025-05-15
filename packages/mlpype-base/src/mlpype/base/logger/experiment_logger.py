"""An interface for logging experiments."""
import json
import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from loguru import logger

from mlpype.base.constants import Constants
from mlpype.base.data import DataSet
from mlpype.base.model.model import Model
from mlpype.base.serialiser import JoblibSerialiser, Serialiser

if TYPE_CHECKING:
    from mlpype.base.experiment import Experiment


class ExperimentLogger(ABC):
    """An interface for logging experiments."""

    @abstractmethod
    def __enter__(self) -> None:
        """Start the experiment."""
        return

    def log_run(
        self,
        exp: "Experiment",
        metrics: Dict[str, Dict[str, Union[str, float, int, str, bool]]],
        transformed: Dict[str, DataSet],
    ) -> None:
        """Log the results of a run.

        This method logs:
            - The metrics for each dataset
            - The pipeline, serialiser, and input/output type checkers
            - The model
            - The plots
            - The requirements.txt

        Args:
            exp (Experiment): The experiment to log
            metrics (Dict[str, Dict[str, Union[str, float, int, str, bool]]]): A dictionary of dicts:
                per dataset: metric names and values
            transformed (Dict[str, DataSet]): A dictionary of transformed datasets (after applying the pipeline)
        """
        for dataset_name, metric_set in metrics.items():
            logger.info(f"Log results: metrics for {dataset_name}")
            self.log_metrics(dataset_name, metric_set)

        self._create_output_folders(exp.output_folder)

        # Log plots, requirements.txt, and extra files
        logger.info("Log plots, requirements.txt, and extra files")
        self._log_plots(exp, transformed)
        self._log_requirements(exp.output_folder)
        self._log_extra_files(exp.output_folder, exp.additional_files_to_store)

        # log serialiser using a JoblibSerialiser
        logger.info("Log results: pipeline, serialiser, input/output type checkers")
        jl_serialiser = JoblibSerialiser()
        self.log_artifact(exp.output_folder / Constants.SERIALISER_FILE, jl_serialiser, object=exp.serialiser)
        self.log_artifact(exp.output_folder / Constants.PIPELINE_FILE, exp.serialiser, object=exp.pipeline)
        self.log_artifact(
            exp.output_folder / Constants.INPUT_TYPE_CHECKER_FILE, exp.serialiser, object=exp.input_type_checker
        )
        self.log_artifact(
            exp.output_folder / Constants.OUTPUT_TYPE_CHECKER_FILE, exp.serialiser, object=exp.output_type_checker
        )

        # Finally, log the model. Do this so mlflow can actually be useful...
        logger.info("Log model")
        self.log_model(exp.model, exp.output_folder / Constants.MODEL_FOLDER)

    def _log_requirements(self, output_folder: Path) -> None:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        version_info = {
            "python_version": python_version,
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
        }
        version_file = output_folder / Constants.PYTHON_VERSION_FILE
        with open(version_file, "w") as f:
            json.dump(version_info, f)
        self.log_file(version_file)

        reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
        requirements_file = output_folder / Constants.REQUIREMENTS_FILE
        with open(requirements_file, "w") as f:
            f.write(reqs)
        self.log_file(requirements_file)

    def _log_extra_files(self, output_folder: Path, additional_files_to_store: List[Union[Path, str]]) -> None:
        """Logs the extra files for an experiment, as specified in the constructor.

        This also supports saving files outside of the current working directory.
        These files are saved under the root name of the file / folder you select.
        """
        paths_to_log = []
        logger.info("Log extra files")
        for extra_file in additional_files_to_store:
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
            target_path = output_folder / logged_path
            self.log_local_file(source_path, target_path)
            paths_to_log.append(logged_path)

        logger.info("Log `extra files`-file")
        extra_files_file = output_folder / Constants.EXTRA_FILES
        with open(extra_files_file, "w") as f:
            data = {"paths": paths_to_log}
            json.dump(data, f)
        # Make sure the file path is closed before logging anything, to make sure all writes have flushed.
        self.log_file(extra_files_file)

    def _log_plots(self, exp: "Experiment", datasets: Dict[str, DataSet]) -> None:
        predicted = {name: exp.model.transform(data) for name, data in datasets.items()}

        full_datasets = {}
        for dataset_name, dataset in predicted.items():
            assert dataset_name in datasets
            transformed_set = datasets[dataset_name]
            for data_name, data in dataset.items():
                new_name = f"{data_name}{Constants.PREDICTION_SUFFIX}"
                assert new_name not in transformed_set
                transformed_set[new_name] = data

            full_datasets[dataset_name] = transformed_set

        for plotter in exp.plots:
            for name, data in full_datasets.items():
                plot_folder = exp.output_folder / Constants.PLOT_FOLDER / name
                plot_folder.mkdir(parents=True, exist_ok=True)
                plot_files = plotter.plot(plot_folder, data, exp)

                for plot_file in plot_files:
                    self.log_file(plot_file)

    def _create_output_folders(self, output_folder: Path) -> None:
        if output_folder.exists():
            logger.warning("Output folder already exists. This may cause conflicts.")
        output_folder.mkdir(exist_ok=True, parents=True)
        (output_folder / Constants.MODEL_FOLDER).mkdir(exist_ok=True)

    @abstractmethod
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Stop the experiment."""
        return

    def log_metrics(self, dataset_name: str, metrics: Dict[str, Union[str, float, int, str, bool]]) -> None:
        """Logs the metrics for a given dataset.

        Metrics will be stored by default as `<dataset_name>_<metric_name>`.

        Args:
            dataset_name (str): The name of the dataset. Will be prepended to any metric name.
            metrics (Dict[str, Union[str, float, int, str, bool]]): A dictionary of metric names and values.
        """
        metrics = {f"{dataset_name}_{name}": metric for name, metric in metrics.items()}
        self._log_metrics(metrics)

    @abstractmethod
    def _log_metrics(self, metrics: Dict[str, Union[str, float, int, str, bool]]) -> None:
        """Perform the actual logging of metrics.

        Args:
            metrics (Dict[str, Union[str, float, int, str, bool]]): A dictionary of metric names and values.
        """
        raise NotImplementedError

    @abstractmethod
    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """Logs the parameters for a given run.

        These will by default be passed on with prefixes such as `model__`.

        Args:
            parameters (Dict[str, Any]): The parameters of a given run. Ideally these
                parameters should be no more complicated than string, float, int, bool, or a
                list of these.
        """
        raise NotImplementedError

    def log_model(self, model: Model, folder: Union[str, Path]) -> None:
        """Logs a Model for a given experiment.

        This function will write the model to the given location as well.

        Args:
            model (Model): The Model to be logged.
            folder (Union[str, Path]): The file to log the Model to.
        """
        model.save(folder)
        self.log_file(folder)

    def log_artifact(self, file: Union[str, Path], serialiser: Serialiser, object: Any) -> None:
        """Logs an artifact/file as part of an experiment.

        Args:
            file (Union[str, Path]): The file to serialise to and subsequently log.
            serialiser (Serialiser): Serialiser used to serialise the given object.
            object (Any): The object to log. This object will be serialised to the given file
                using the serialiser and logged using `log_file`
        """
        serialiser.serialise(object, file)
        self.log_file(file)

    def log_local_file(self, file: Union[str, Path], output_target: Union[str, Path]) -> None:
        """Logs a given local file / directory as part of an experiment.

        First the file / directory is copied to the output directory, then
        log_file is called to make it part of the experiment.

        Args:
            file (Union[str, Path]): The file / directory to log.
            output_target (Union[str, Path]): The target location of the file / directory.
        """
        file = Path(file)
        output_target = Path(output_target)
        output_target.parent.mkdir(exist_ok=True, parents=True)

        if file.is_file():
            shutil.copy(file, output_target)
        else:
            if output_target.exists():
                shutil.rmtree(output_target)
            shutil.copytree(file, output_target)

        self.log_file(output_target)

    @abstractmethod
    def log_file(self, file: Union[str, Path]) -> None:
        """Logs a given file as part of an experiment.

        Args:
            file (Union[str, Path]): The file to log.
        """
        raise NotImplementedError
