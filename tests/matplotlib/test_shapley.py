from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
from pytest import fixture, mark

from mlpype.base.data.dataset import DataSet
from mlpype.matplotlib.evaluate.shapley import ShapleyPlot
from mlpype.sklearn.model import LinearRegressionModel


@fixture
def shapley_plot() -> ShapleyPlot:
    return ShapleyPlot("x", "y")


def test_integration():
    with TemporaryDirectory() as tmp_dir:
        folder = Path(tmp_dir) / "plots"
        ds = DataSet(x=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), y=np.array([3, 6, 9]))
        model = LinearRegressionModel(inputs=["x"], outputs=["y"])
        model.fit(ds)

        experiment = MagicMock()
        experiment.model = model

        shapley_plot = ShapleyPlot("x", "y")

        files = shapley_plot.plot(folder, ds, experiment)

        for f in files:
            assert f.exists()


def test_plot(shapley_plot: ShapleyPlot):
    folder = Path("someroot")
    dataset = DataSet(x=[1, 2, 3])
    experiment = MagicMock()
    sampled = MagicMock()
    shap = MagicMock()

    partial_plots = [1, 2, 3]
    swarm_plot = 4

    with patch.object(shapley_plot, "_sample_and_shapley", return_value=(sampled, shap)) as mock_sample, patch.object(
        shapley_plot, "_partial_plots", return_value=partial_plots
    ) as mock_partial, patch.object(
        shapley_plot, "_beeswarm_plot", return_value=swarm_plot
    ) as mock_swarm, patch.object(
        shapley_plot, "_make_forecast_function"
    ) as mock_forecast, patch(
        "pathlib.Path.mkdir"
    ) as mock_mkdir:
        result = shapley_plot.plot(folder, dataset, experiment)

    root_folder = folder / "shapley"
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_forecast.assert_called_once_with(experiment.model)
    mock_sample.assert_called_once_with(dataset, mock_forecast.return_value)
    mock_partial.assert_called_once_with(root_folder, mock_forecast.return_value, sampled)
    mock_swarm.assert_called_once_with(root_folder, shap)

    assert result == partial_plots + [swarm_plot]


@mark.parametrize(
    ["data"],
    [
        [np.array([[1, 2], [3, 4], [5, 6]])],
        [pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})],
    ],
)
def test_sample_df_or_np(shapley_plot: ShapleyPlot, data: Union[np.ndarray, pd.DataFrame]):
    sample = shapley_plot._sample_df_or_np(data, 1)

    assert sample.shape == (1, data.shape[1])


def test_make_forecast_function(shapley_plot: ShapleyPlot):
    output = [4, 5, 6]
    model = MagicMock()
    model.transform.return_value = {"y": output}
    data = [1, 2, 3]

    fnc = shapley_plot._make_forecast_function(model)
    result = fnc(data)

    model.transform.assert_called_once_with(DataSet({"x": [1, 2, 3]}))
    assert result == output


def test_sample_and_shapley():
    sampler = MagicMock()
    plotter = ShapleyPlot("x", "y", sample_size=10, sample_function=sampler)
    model = MagicMock()
    data = DataSet(x=[1, 2, 3])

    with patch("mlpype.matplotlib.evaluate.shapley.Explainer") as mock_explainer:
        sampled, shapley = plotter._sample_and_shapley(data, model)

    sampler.assert_called_once_with([1, 2, 3], 10)
    mock_explainer.assert_called_once_with(model, sampler.return_value)
    mock_explainer.return_value.assert_called_once_with(sampler.return_value)
    shapley_values = mock_explainer.return_value.return_value

    assert sampled == sampler.return_value
    assert shapley == shapley_values


@mark.parametrize(
    ["data", "indices"],
    [
        [np.array([[1, 2], [3, 4], [5, 6]]), [0, 1]],
        [pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), ["a", "b"]],
    ],
)
def test_partial_plots(shapley_plot: ShapleyPlot, data: Union[np.ndarray, pd.DataFrame], indices: list):
    root_folder = Path("great_folder")
    model = MagicMock()

    with patch("mlpype.matplotlib.evaluate.shapley.partial_dependence_plot") as mock_partial_dependence_plot, patch(
        "mlpype.matplotlib.evaluate.shapley.plt"
    ) as mock_plt, patch("pathlib.Path.mkdir") as mock_mkdir:
        result = shapley_plot._partial_plots(root_folder, model, data)

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_partial_dependence_plot.assert_has_calls([call(i, model, data, show=False) for i in indices])
    mock_plt.tight_layout.assert_has_calls([call() for _ in indices])
    mock_plt.savefig.assert_has_calls([call(root_folder / "partial_dependence" / f"{i}.png") for i in indices])
    mock_plt.close.assert_has_calls([call() for _ in indices])

    assert len(result) == len(indices)
    assert result == [root_folder / "partial_dependence" / f"{i}.png" for i in indices]


def test_beeswarm_plot(shapley_plot: ShapleyPlot):
    shapley = MagicMock()
    root_folder = Path("great_folder")

    with patch("mlpype.matplotlib.evaluate.shapley.plots.beeswarm") as mock_beeswarm, patch(
        "mlpype.matplotlib.evaluate.shapley.plt"
    ) as mock_plt:
        result = shapley_plot._beeswarm_plot(root_folder, shapley)

    mock_beeswarm.assert_called_once_with(shapley, show=False)
    mock_plt.tight_layout.assert_called_once_with()
    mock_plt.savefig.assert_called_once_with(result)
    mock_plt.close.assert_called_once_with()

    assert isinstance(result, Path)
    assert result == root_folder / "beeswarm.png"
