"""Provides tools to simplify making Shapley plots in mlpype."""
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from shap import Explainer, partial_dependence_plot, plots

from mlpype.base.data.dataset import DataSet
from mlpype.base.evaluate.plot import BasePlotter
from mlpype.base.experiment.experiment import Experiment
from mlpype.base.model.model import Model
from mlpype.sklearn.data.sklearn_data import SklearnData

SkleanData = Union[np.ndarray, pd.DataFrame]


class ShapleyPlot(BasePlotter):
    """Generates Shapley plots for the a model and dataset."""

    def __init__(
        self,
        input_name: str,
        output_name: str,
        sample_size: int = 100,
        sample_function: Optional[Callable[[Any, int], SkleanData]] = None,
    ):
        """
        A plotter for shapley values.

        This creates:
            - Partial dependence plots for each feature.
            - Beeswarm plots for the dataset.

        This will be run on a sampled dataset. Currently, only 1 input/output dataset is supported. By
        default, Shapley only works for numpy arrays and pandas dataframes. We do not impose this
        restriction, and leave it to the user to try and make it work for other types of data.

        Args:
            input_name (str): The name of the input dataset.
            output_name (str): The name of the output dataset.
            sample_size (int, optional): The size of the sampled dataset. Shapley values are expensive to
                compute, and can be slow. To minimise the impact of this, we sample the dataset.
                If the dataset is larger than the sample size, we use the full dataset. Defaults to 100.
            sample_function (Optional[Callable[[Any, int], SkleanData]], optional): A function to
                subsample the input dataset. By default, `_sample_df_or_np` of this object is used.
                This only supports numpy arrays and pandas dataframes.
        """
        super().__init__()
        self.input_name = input_name
        self.output_name = output_name

        if sample_function is None:
            sample_function = self._sample_df_or_np

        self.sample_function = sample_function
        self.sample_size = sample_size

    def _sample_df_or_np(self, data: SkleanData, sample_size: int) -> SkleanData:
        if isinstance(data, np.ndarray):
            if data.shape[0] >= sample_size:
                return data[np.random.choice(data.shape[0], sample_size)]
        elif isinstance(data, pd.DataFrame):
            if data.shape[0] >= sample_size:
                return data.sample(n=sample_size)
        else:
            raise ValueError(f"Input data must be either a numpy array or a pandas dataframe. Got {type(data)}")

        logger.warning(f"Not enough data to sample. Using full dataset. Got {data.shape[0]}")
        return data

    def plot(self, plot_folder: Path, data: DataSet, experiment: Experiment) -> List[Path]:
        """Creates plots and writes them to `plot_folder / "shapley"`.

        Args:
            plot_folder (Path): The folder to write the plot to. The final plot will be written to
                `plot_folder / "shapley"`.
            data (DataSet): A DataSet containing all the data you need to make your plots.
                In an Experiment, this will contain the last DataSet from the Pipeline with the
                predictions added as "{output_name}{Constants.PREDICTION_POSTFIX}"
            experiment (Experiment): The experiment object. Used to extract the model.

        Returns:
            List[Path]: The file paths of where the plot(s) are stored.
        """
        root_folder = plot_folder / "shapley"
        root_folder.mkdir(parents=True, exist_ok=True)

        forecast = self._make_forecast_function(experiment.model)
        sampled, shap_values = self._sample_and_shapley(data, forecast)

        result = []
        result += self._partial_plots(root_folder, forecast, sampled)
        result.append(self._beeswarm_plot(root_folder, shap_values))

        return result

    def _make_forecast_function(self, model: Model) -> Callable[[SklearnData], Any]:
        def forecast(X: SkleanData) -> SkleanData:
            return model.transform(DataSet(x=X))[self.output_name]

        return forecast

    def _sample_and_shapley(self, data: DataSet, model: Callable) -> Tuple[SkleanData, np.ndarray]:
        """
        Samples the input data and returns the shapley values.

        Args:
            data (DataSet): The input Dataset.
            model (Callable): The model-predict function.

        Returns:
            Tuple[SkleanData, np.ndarray]: The sampled data and the shapley values.
        """
        input_data = data.get(self.input_name)
        if not isinstance(input_data, (np.ndarray, pd.DataFrame)):
            logger.warning("Input data is not an array or dataframe. Shapley may not work.")

        # Shapley values on sampled data
        sampled = self.sample_function(input_data, self.sample_size)
        expl = Explainer(model, sampled)
        shap_values = expl(sampled)

        return sampled, shap_values

    def _partial_plots(self, root_folder: Path, model: Callable, sampled: SkleanData) -> List[Path]:
        """Plots partial dependency plots for all column."""
        if isinstance(sampled, np.ndarray):
            column_ids = list(range(sampled.shape[1]))
        elif isinstance(sampled, pd.DataFrame):
            column_ids = list(sampled.columns)
        else:
            raise ValueError(f"Input data must be either a numpy array or a pandas dataframe. Got {type(sampled)}")

        partial_deps_folder = root_folder / "partial_dependence"
        partial_deps_folder.mkdir(parents=True, exist_ok=True)

        result = []
        for col in column_ids:
            logger.info(f"Plotting partial dependence for column: {col}")
            partial_plot_file = partial_deps_folder / f"{col}.png"
            partial_dependence_plot(col, model, sampled, show=False)
            plt.tight_layout()
            plt.savefig(partial_plot_file)
            plt.close()
            result.append(partial_plot_file)
        return result

    def _beeswarm_plot(self, root_folder: Path, shap_values: np.ndarray) -> Path:
        """Makes a beeswarm plot."""
        logger.info("Plotting beeswarm plot")
        beeplot_file = root_folder / "beeswarm.png"

        plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(beeplot_file)
        plt.close()

        return beeplot_file
