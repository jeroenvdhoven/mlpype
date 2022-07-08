from pathlib import Path
from typing import Any

from .experiment_logger import ExperimentLogger


class LocalLogger(ExperimentLogger):
    def __init__(self) -> None:
        """A basic logger which only saves local copies of the data and prints results."""
        super().__init__()

    def _log_metrics(self, metrics: dict[str, float | int | str | bool]) -> None:
        """Prints the metrics to the console.

        Args:
            metrics (dict[str, float | int | str | bool]): A dictionary of metric names and values.
        """
        for name, value in metrics.items():
            print(f"{name.ljust(20)}: {value}")

    def log_parameters(self, parameters: dict[str, Any]) -> None:
        """Prints the parameters to the console.

        Args:
            parameters (dict[str, Any]): The parameters of a given run. Ideally these
                parameters should be no more complicated than string, float, int, bool, or a
                list of these.
        """
        for name, value in parameters.items():
            print(f"{name.ljust(20)}: {value}")

    def log_file(self, file: str | Path) -> None:
        """This method does no additional logging.

        Args:
            file (str | Path): The file to log. No actual logging is done.
        """