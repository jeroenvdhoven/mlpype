from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Optional, Type, Union

from .experiment_logger import ExperimentLogger


class LocalLogger(ExperimentLogger):
    def __enter__(self) -> None:
        """Start the experiment."""

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        """Stop the experiment."""

    def __init__(self, min_whitespace: int = 30) -> None:
        """A basic logger which only saves local copies of the data and prints results."""
        super().__init__()
        self.min_whitespace = min_whitespace

    def _log_metrics(self, metrics: Dict[str, Union[str, float, int, str, bool]]) -> None:
        """Prints the metrics to the console.

        Args:
            metrics (Dict[str, Union[str, float, int, str, bool]]): A dictionary of metric names and values.
        """
        for name, value in metrics.items():
            print(f"{name.ljust(self.min_whitespace)}: {value}")

    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """Prints the parameters to the console.

        Args:
            parameters (Dict[str, Any]): The parameters of a given run. Ideally these
                parameters should be no more complicated than string, float, int, bool, or a
                list of these.
        """
        for name, value in parameters.items():
            print(f"{name.ljust(self.min_whitespace)}: {value}")

    def log_file(self, file: Union[str, Path]) -> None:
        """This method does no additional logging.

        Args:
            file (Union[str, Path]): The file to log.
        """
