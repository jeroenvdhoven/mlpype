from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Union

from pype.base.model.model import Model
from pype.base.serialiser import Serialiser


class ExperimentLogger(ABC):
    # TODO: make a LocalLogger.

    def __enter__(self) -> None:
        """Start the experiment."""
        return

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop the experiment."""
        return

    def log_metrics(self, dataset_name: str, metrics: Dict[str, Union[float, int, str | bool]]) -> None:
        """Logs the metrics for a given dataset.

        Metrics will be stored by default as `<dataset_name>_<metric_name>`.

        Args:
            dataset_name (str): The name of the dataset. Will be prepended to any metric name.
            metrics (Dict[str, Union[float, int, str  |  bool]]): A dictionary of metric names and values.
        """
        metrics = {f"{dataset_name}_{name}": metric for name, metric in metrics.items()}
        self._log_metrics(metrics)

    @abstractmethod
    def _log_metrics(self, metrics: dict[str, float | int | str | bool]) -> None:
        """Perform the actual logging of metrics.

        Args:
            metrics (dict[str, float | int | str | bool]): A dictionary of metric names and values.
        """
        raise NotImplementedError

    @abstractmethod
    def log_parameters(self, parameters: dict[str, Any]) -> None:
        """Logs the parameters for a given run.

        These will by default be passed on with prefixes such as `model__`.

        Args:
            parameters (dict[str, Any]): The parameters of a given run. Ideally these
                parameters should be no more complicated than string, float, int, bool, or a
                list of these.
        """
        raise NotImplementedError

    def log_model(self, model: Model, folder: str | Path) -> None:
        """Logs a Model for a given experiment.

        This function will write the model to the given location as well.

        Args:
            model (Model): The Model to be logged.
            MODEL_FOLDER (str | Path): The file to log the Model to.
        """
        model.save(folder)
        self.log_file(folder)

    def log_artifact(self, file: str | Path, serialiser: Serialiser, object: Any) -> None:
        """Logs an artifact/file as part of an experiment.

        Args:
            file (str | Path): The file to serialise to and subsequently log.
            serialiser (Serialiser): Serialiser used to serialise the given object, if any.
            object (Any): The object to log. This object will be serialised to the given file
                using the serialiser and logged using `log_file`
        """
        if object is not None and serialiser is not None:
            serialiser.serialise(object, file)
        self.log_file(file)

    @abstractmethod
    def log_file(self, file: str | Path) -> None:
        """Logs a given file as part of an experiment.

        Args:
            file (str | Path): The file to log.
        """
        raise NotImplementedError
