from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pytest import mark

from mlpype.base.data.dataset import DataSet
from mlpype.light_gbm.model import LightGBMModel


@mark.parametrize(["as_pandas"], [[True], [False]])
def test_lightgbm_model_serialisation(as_pandas: bool):
    nrows = 50
    nfeat = 20
    np.random.seed(1)
    x = np.random.normal(0, 1, (nrows, nfeat))
    y = (x.sum(axis=1) > 0.5).astype(int)

    if as_pandas:
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)

    ds = DataSet(x=x, y=y)
    model = LightGBMModel(inputs=["x"], outputs=["y"], num_trees=10, num_threads=1, objective="binary")

    model.fit(ds)
    result = model.transform(ds)

    assert result["y"].shape == (nrows,)

    with TemporaryDirectory() as td:
        td = Path(td)
        file = td / "tmp.bin"
        model.save(file)
        model = LightGBMModel.load(file)

        new_pred = model.transform(ds)
        assert new_pred["y"].shape == (nrows,)
        np.testing.assert_array_equal(new_pred["y"], result["y"])


@mark.parametrize(["as_pandas"], [[True], [False]])
def test_lightgbm_model_training(as_pandas: bool):
    nrows = 50
    nfeat = 20
    np.random.seed(1)
    x = np.random.normal(0, 1, (nrows, nfeat))
    y = (x.sum(axis=1) > 0.5).astype(int)

    if as_pandas:
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)

    ds = DataSet(x=x, y=y)
    model = LightGBMModel(inputs=["x"], outputs=["y"], num_trees=10, num_threads=1, objective="binary")

    model.fit(ds)
    result = model.transform(ds)

    assert result["y"].shape == (nrows,)
