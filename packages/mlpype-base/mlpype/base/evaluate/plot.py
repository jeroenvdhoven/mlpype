"""Contains tools to create plots for mlpype experiments."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Protocol, Union

from mlpype.base.data.dataset import DataSet


class PlotFunction(Protocol):
    """A function definition for creating plots from data."""

    def __call__(self, path: Path, *data: Any) -> None:
        """Creates a plot from the given data and writes it to the given path.

        Args:
            path (Path): The path to save the plot to.
            *data (Any): Datasets to be used in plotting.
        """


class BasePlotter(ABC):
    """A base class for creating plots."""

    @abstractmethod
    def plot(self, plot_folder: Path, data: DataSet, experiment: Any) -> List[Path]:
        """Creates a plot from the given data and writes it to the given path.

        Args:
            plot_folder (Path): The folder to write the plot to. You still need to set your
                file name. If multiple plots are created, you can use the same folder, but
                make sure to return all file names in the list.
            data (DataSet): The full dataset to plot. This should contain all the data you
                need to make your plots. It contains the last DataSet from the pipeline,
                with the predictions added as "{output_name}{Constants.PREDICTION_POSTFIX}"
            experiment (Any): The experiment object. Useful for extracting
                the trained model.

        Returns:
            List[Path]: The file paths of where the plot(s) are stored.
        """
        raise NotImplementedError


@dataclass
class Plotter(BasePlotter):
    """Creates 1 plot from data."""

    plot_function: PlotFunction
    file_name: Union[Path, str]
    dataset_names: List[str]

    def plot(self, plot_folder: Path, data: DataSet, experiment: Any) -> List[Path]:
        """Selects the correct data from the DataSet, creates a plot, and writes it to `plot_folder / self.file_name`.

        This is a simple wrapper to make a single plot.

        Args:
            plot_folder (Path): The folder to write the plot to. The final plot will be written to
                `plot_folder / self.file_name`.
            data (DataSet): A DataSet containing all the data you need to make your plots.
                In an Experiment, this will contain the last DataSet from the Pipeline with the
                predictions added as "{output_name}{Constants.PREDICTION_POSTFIX}"
            experiment (Any): The experiment object. Not used by default for this
                plotter; it assumens you only use the transformed input data.

        Returns:
            List[Path]: The file paths of where the plot(s) are stored.
        """
        plot_ds = data.get_all(self.dataset_names)
        plot_file = plot_folder / self.file_name
        self.plot_function(plot_file, *plot_ds)
        return [plot_file]
