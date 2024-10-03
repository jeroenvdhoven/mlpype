from pathlib import Path
from unittest.mock import MagicMock

from mlpype.base.evaluate.plot import Plotter


def test_Plotter():
    fn = "filename"
    plot_func = MagicMock()
    names = ["foo", "bar", "baz"]

    dataset = MagicMock()
    data = [1, 2, 3]
    dataset.get_all.return_value = data

    plotter = Plotter(
        plot_function=plot_func,
        file_name=fn,
        dataset_names=names,
    )

    tmp_dir = Path("tempdir")
    result = plotter.plot(tmp_dir, dataset, MagicMock())

    dataset.get_all.assert_called_once_with(names)
    plot_func.assert_called_once_with(tmp_dir / fn, *data)
    assert result == [tmp_dir / fn]
