import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pytest import fixture

from mlpype.base.data.dataset import DataSet
from mlpype.xgboost.model import XGBRegressorModel


@fixture
def dataset() -> DataSet:
    x = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [9, 3, 2, 3, 8],
        }
    )

    y = pd.DataFrame(
        {
            "target": [11, 2, 3, 4, 15],
        }
    )
    return DataSet(x=x, y=y)


@fixture
def model() -> XGBRegressorModel:
    return XGBRegressorModel(
        inputs=["x"],
        outputs=["y"],
        n_estimators=20,
        objective="reg:squarederror",
        max_depth=3,
    )


def error(target, prediction):
    return float((target - prediction).abs().sum())


def test_xgb_regressor_training(dataset: DataSet, model: XGBRegressorModel):
    # training works if trained model produces better performance than mean

    baseline = error(dataset["y"]["target"], dataset["y"]["target"].mean())

    model.fit(dataset)

    predictions = model.transform(dataset)["y"]
    performance = error(dataset["y"]["target"], predictions)

    assert performance < baseline


def test_xgb_regressor_saving_and_loading(dataset: DataSet, model: XGBRegressorModel):
    # test if we can save and load the model
    target_path = Path("tmp_xgboost")

    try:
        model.fit(dataset)
        predictions_old = model.transform(dataset)["y"]

        model.save(target_path)
        new_model = XGBRegressorModel.load(target_path)
        predictions_new = new_model.transform(dataset)["y"]

        np.testing.assert_array_equal(predictions_old, predictions_new)
    finally:
        shutil.rmtree(target_path)
    assert not target_path.exists()
