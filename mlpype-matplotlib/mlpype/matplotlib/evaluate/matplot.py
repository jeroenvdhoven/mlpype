"""Provides tools to simplify plotting using matplotlib in mlpype."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Protocol, Union

import matplotlib.pyplot as plt

from mlpype.base.data.dataset import DataSet
from mlpype.base.evaluate.plot import BasePlotter
from mlpype.base.experiment.experiment import Experiment


class MatplotFunction(Protocol):
    """Protocol for creating plots using matplotlib."""

    def __call__(self, *data: Any) -> None:
        """Creates a plot from the given data.

        Saving the plot isn't handled by this function.

        Args:
            *data (Any): Datasets to be used in plotting.
        """


@dataclass
class MatplotlibPlotter(BasePlotter):
    """Creates plots using matplotlib."""

    plot_function: MatplotFunction
    file_name: Union[Path, str]
    dataset_names: List[str]

    def plot(self, plot_folder: Path, data: DataSet, experiment: Experiment) -> List[Path]:
        """Creates 1 plot from the given dataset and writes it to the given path using matplotlib.

        This is a simple wrapper to make a single matplotlib plot.

        Args:
            plot_folder (Path): The folder to write the plot to. The final plot will be written to
                `plot_folder / self.file_name`.
            data (DataSet): A DataSet containing all the data you need to make your plots.
                In an Experiment, this will contain the last DataSet from the Pipeline with the
                predictions added as "{output_name}{Constants.PREDICTION_POSTFIX}"
            experiment (Experiment): The experiment object. Not used by default for this
                plotter; it assumens you only use the transformed input data.

        Returns:
            List[Path]: The file paths of where the plot(s) are stored.
        """
        plot_path = plot_folder / self.file_name
        self.plot_function(*data.get_all(self.dataset_names))
        plt.savefig(plot_path)
        plt.close()
        return [plot_path]
