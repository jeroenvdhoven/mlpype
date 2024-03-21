import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Optional, Type, Union

from mlpype.base.model.model import Model
from mlpype.base.serialiser import Serialiser


class ExperimentLogger(ABC):
    @abstractmethod
    def __enter__(self) -> None:
        """Start the experiment."""
        return

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
            MODEL_FOLDER (Union[str, Path]): The file to log the Model to.
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
