from pathlib import Path
from typing import Iterable, List, Union
from unittest.mock import patch

from pytest import fixture

from mlpype.base.data.dataset import DataSet
from mlpype.base.model.model import Model


class DummyModel(Model[int]):
    def __init__(self, seed: int, inputs: List[str], outputs: List[str]) -> None:
        super().__init__(inputs=inputs, outputs=outputs, seed=seed)
        self._fitted = False

    def _transform(self, *data: int) -> Iterable[int]:
        return [i + 1 for i in data]

    def _fit(self, *data: int) -> None:
        self._fitted = True

    def set_seed(self) -> None:
        pass

    def _save(self, file: Union[str, Path]) -> None:
        pass

    @classmethod
    def _load(cls):
        pass


class Test_Model:
    @fixture
    def model(self):
        return DummyModel(1, ["in"], ["out"])

    @fixture
    def multi_model(self):
        return DummyModel(1, ["in", "in2"], ["out", "out2"])

    @fixture
    def data(self):
        return DataSet[int](
            {
                "in": 3,
                "out": 5,
            }
        )

    @fixture
    def multi_data(self):
        return DataSet[int]({"in": 3, "in2": 4, "out": 5, "out2": 6})

    def test_fit(self, model: DummyModel, data: DataSet[int]):
        with patch.object(model, "_fit") as mock_fit:
            result = model.fit(data)

            assert result == model
            mock_fit.assert_called_once_with(3, 5)

    def test_multi_fit(self, multi_model: DummyModel, multi_data: DataSet[int]):
        with patch.object(multi_model, "_fit") as mock_fit:
            result = multi_model.fit(multi_data)

            assert result == multi_model
            mock_fit.assert_called_once_with(3, 4, 5, 6)

    def test_transform(self, model: DummyModel, data: DataSet[int]):
        with patch.object(model, "_transform") as mock_transform:
            result = model.transform(data)

            mock_transform.assert_called_once_with(3)
            assert isinstance(result, DataSet)
            assert len(result) == 1
            assert result["out"] == mock_transform.return_value

    def test_transform_without_outputs(self, model: DummyModel, data: DataSet[int]):
        del data["out"]
        with patch.object(model, "_transform") as mock_transform:
            result = model.transform(data)

            mock_transform.assert_called_once_with(3)
            assert isinstance(result, DataSet)
            assert len(result) == 1
            assert result["out"] == mock_transform.return_value

    def test_multi_transform(self, multi_model: DummyModel, multi_data: DataSet[int]):
        preds = [9, 10]
        with patch.object(multi_model, "_transform", return_value=preds) as mock_transform:
            result = multi_model.transform(multi_data)

            mock_transform.assert_called_once_with(3, 4)
            assert isinstance(result, DataSet)
            assert len(result) == 2
            assert result["out"] == preds[0]
            assert result["out2"] == preds[1]
