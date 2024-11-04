from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List

from pytest import fixture

from mlpype.base.data.dataset import DataSet
from mlpype.base.model.hierarchical_model import HierarchicalModel
from mlpype.base.model.model import Model


class DummyModel(Model[List[int]]):
    def __init__(self, seed: int, inputs: List[str], outputs: List[str]) -> None:
        super().__init__(inputs=inputs, outputs=outputs, seed=seed)
        self.mean = None

    def _transform(self, data: List[int]) -> List[int]:
        assert self.mean is not None, "Mean not set"
        return [self.mean for _ in data]

    def _fit(self, _: List[int], y: List[int]) -> None:
        self.mean = int(sum(y) / len(y))

    def set_seed(self) -> None:
        pass

    def _save(self, folder: Path) -> None:
        with open(folder / "mean", "w") as f:
            f.write(str(self.mean))

    @classmethod
    def _load(cls, folder: Path, inputs: List[str], outputs: List[str]) -> "DummyModel":
        with open(folder / "mean", "r") as f:
            mean = int(f.read())
        res = cls(inputs=inputs, outputs=outputs, seed=1)
        res.mean = mean
        return res


class HierarchicalDummyModel(HierarchicalModel[DummyModel]):
    pass


def splitter(x: List[int]) -> Dict[str, List[int]]:
    mid = int(len(x) / 2)
    return {"low": x[:mid], "high": x[mid:]}


def merger(x: Dict[str, List[int]]) -> List[int]:
    return x["low"] + x["high"]


class TestHierarchicalModel:
    @fixture
    def model(self) -> HierarchicalDummyModel:
        return HierarchicalDummyModel(["in"], ["out"], data_splitter=splitter, data_merger=merger)

    @fixture
    def data(self) -> DataSet[List[int]]:
        return DataSet[List[int]](
            {
                "in": [3, 4, 5, 6, 7, 8, 9, 10],
                "out": [5, 6, 7, 8, 9, 10, 11, 12],
            }
        )

    def test_fit_transform_integration(self, model: HierarchicalDummyModel, data: DataSet[List[int]]):
        result = model.fit(data)
        assert result == model
        for m in model.model.values():
            assert m.mean is not None

        result = model.transform(data)
        assert len(result["out"]) == len(data["in"])

        assert result == {
            "out": [*[model.model["low"].mean for _ in range(4)], *[model.model["high"].mean for _ in range(4)]]
        }

    def test_save_load_integration(self, model: HierarchicalDummyModel, data: DataSet[List[int]]):
        model.fit(data)

        with TemporaryDirectory() as tmp_dir:
            tmp_file = Path(tmp_dir) / "test"
            model.save(tmp_file)
            loaded_model = HierarchicalDummyModel.load(tmp_file)
        assert isinstance(loaded_model, HierarchicalDummyModel)

        assert len(model.model) == len(loaded_model.model)
        for name, submodel in model.model.items():
            assert name in loaded_model.model
            assert submodel.mean == loaded_model.model[name].mean

        result = model.transform(data)
        loaded_result = loaded_model.transform(data)

        assert result == loaded_result

    def test_class_from_sklearn_model_class(self):
        klass = HierarchicalModel.class_from_model_class(DummyModel)

        assert klass.__name__ == "HierarchicalDummyModel"
        assert issubclass(klass, HierarchicalModel)

        annotated = klass._get_annotated_class()
        assert annotated == DummyModel

    def test_from_sklearn_model_class(self, data: DataSet[List[int]]):
        model = HierarchicalModel.from_model_class(DummyModel, ["in"], ["out"], splitter, merger)

        assert model.__class__.__name__ == "HierarchicalDummyModel"
        assert isinstance(model, HierarchicalModel)

        annotated = model._get_annotated_class()
        assert annotated == DummyModel

        model.fit(data)

        predictions = model.transform(data)

        assert len(predictions["out"]) == len(data["in"])
        assert predictions == {
            "out": [*[model.model["low"].mean for _ in range(4)], *[model.model["high"].mean for _ in range(4)]]
        }
