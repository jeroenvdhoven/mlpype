from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pandas as pd

from mlpype.base.data.dataset import DataSet
from mlpype.matplotlib.evaluate.matplot import MatplotlibPlotter


def plot(x: pd.DataFrame):
    x.hist("x", bins=24)


class TestMatplotlibPlotter:
    def test_unit(self):
        fn = "filename"
        plot_func = MagicMock()
        names = ["foo", "bar", "baz"]

        dataset = MagicMock()
        data = [1, 2, 3]
        dataset.get_all.return_value = data

        plotter = MatplotlibPlotter(
            plot_function=plot_func,
            file_name=fn,
            dataset_names=names,
        )

        tmp_dir = Path("tempdir")
        with patch("mlpype.matplotlib.evaluate.matplot.plt") as mock_matplotlib:
            result = plotter.plot(tmp_dir, dataset, MagicMock())

        mock_matplotlib.savefig.assert_called_once_with(tmp_dir / fn)
        mock_matplotlib.close.assert_called_once_with()

        dataset.get_all.assert_called_once_with(names)
        plot_func.assert_called_once_with(*data)
        assert result == [tmp_dir / fn]

    def test_integration(self):
        fn = "filename.png"
        names = ["df"]

        dataset = DataSet(
            df=pd.DataFrame({"x": [1, 2, 3]}),
        )

        plotter = MatplotlibPlotter(
            plot_function=plot,
            file_name=fn,
            dataset_names=names,
        )

        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            file_list = plotter.plot(tmp_dir, dataset, MagicMock())

            assert len(file_list) == 1
            assert file_list[0].exists()
