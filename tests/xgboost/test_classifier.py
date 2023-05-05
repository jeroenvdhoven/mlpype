import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pytest import fixture

from mlpype.base.data.dataset import DataSet
from mlpype.xgboost.model import XGBClassifierModel


@fixture
def dataset() -> DataSet:
    n = 3
    x = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5] * n,
            "b": [20, 3, 2, 3, 20] * n,
        }
    )

    y = pd.DataFrame(
        {
            "target": [1, 0, 0, 0, 1] * n,
        }
    )
    return DataSet(x=x, y=y)


@fixture
def model() -> XGBClassifierModel:
    return XGBClassifierModel(
        inputs=["x"],
        outputs=["y"],
        n_estimators=10,
        max_depth=3,
        learning_rate=0.01,
        min_child_weight=1,  # Please do not change this. Weird bugs may happen
    )


def error(target, prediction):
    return float((target == prediction).mean())


def test_xgb_regressor_training(dataset: DataSet, model: XGBClassifierModel):
    # training works if trained model produces better performance than mean

    baseline = error(dataset["y"]["target"], 0)

    model.fit(dataset)

    predictions = model.transform(dataset)["y"]
    performance = error(dataset["y"]["target"], predictions)

    assert performance > baseline


def test_xgb_regressor_saving_and_loading(dataset: DataSet, model: XGBClassifierModel):
    # test if we can save and load the model
    target_path = Path("tmp_xgboost")

    try:
        model.fit(dataset)
        predictions_old = model.transform(dataset)["y"]

        model.save(target_path)
        new_model = XGBClassifierModel.load(target_path)
        predictions_new = new_model.transform(dataset)["y"]

        np.testing.assert_array_equal(predictions_old, predictions_new)
    finally:
        shutil.rmtree(target_path)
    assert not target_path.exists()
