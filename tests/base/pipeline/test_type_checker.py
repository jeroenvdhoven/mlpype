from typing import List, Type
from unittest.mock import call, patch

from pydantic import create_model
from pytest import fixture

from mlpype.base.data.dataset import DataSet
from mlpype.base.pipeline.type_checker import DataModel, DataSetModel, DataSetTypeChecker, TypeCheckerPipe
from tests.utils import DummyDataModel, DummyTypeChecker, get_dummy_data, pytest_assert


class DummyDataStr(DataModel):
    data: List[str]

    def convert(self) -> List[str]:
        return self.data

    @classmethod
    def to_model(cls, data: List[str]) -> "DataModel":
        return cls(data=data)


class Test_DataSetModel:
    @fixture
    def ModelClass(self) -> Type[DataSetModel]:
        return create_model(
            "DummyDataSetModel", floats=(DummyDataModel, ...), strs=(DummyDataStr, ...), __base__=DataSetModel
        )

    def test_convert(self, ModelClass: Type[DataSetModel]):
        model = ModelClass(
            floats=DummyDataModel(data=[1.0, 3, 4]),
            strs=DummyDataStr(data=["a", "b"]),
        )

        result = model.convert()

        assert result["floats"] == [1.0, 3, 4]
        assert result["strs"] == ["a", "b"]

    def test_to_model(self, ModelClass: Type[DataSetModel]):
        dataset = DataSet(floats=[1.0, 3, 4], strs=["a", "b"])
        result = ModelClass.to_model(dataset)

        expected = ModelClass(
            floats=DummyDataModel(data=[1.0, 3, 4]),
            strs=DummyDataStr(data=["a", "b"]),
        )

        assert expected == result


class Test_TypeCheckerPipe:
    def test_get_pydantic_types(self):
        pipe = TypeCheckerPipe(
            "inputs",
            ["x"],
            [],
        )

        with patch.object(pipe.operator, "get_pydantic_types") as mock_get:
            result = pipe.get_pydantic_types()
            assert result == mock_get.return_value

            mock_get.assert_called_once_with(["x"])

    def test_get_pydantic_types_with_names(self):
        pipe = TypeCheckerPipe(
            "inputs",
            ["x"],
            [],
        )

        with patch.object(pipe.operator, "get_pydantic_types") as mock_get:
            names = ["a", "b", "c"]
            result = pipe.get_pydantic_types(names)
            assert result == mock_get.return_value

            mock_get.assert_called_once_with(names)


class Test_DataSetTypeChecker:
    @fixture
    def type_checker(self, data: DataSet) -> DataSetTypeChecker:
        return DataSetTypeChecker(input_names=list(data.keys()), type_checker_classes=[DummyTypeChecker])

    @fixture
    def data(self) -> DataSet:
        return get_dummy_data(10, 2, 3).read()

    def test_fit(self, type_checker: DataSetTypeChecker, data: DataSet):
        with patch.object(DummyTypeChecker, "fit") as mock_fit:
            type_checker.fit(data["x"], data["y"])

            mock_fit.assert_has_calls([call(data["x"]), call(data["y"])])

            for name in data.keys():
                assert name in type_checker.type_checkers
                assert isinstance(type_checker.type_checkers[name], DummyTypeChecker)

    def test_fit_warning(self, type_checker: DataSetTypeChecker):
        dummy_data = {"z": 1}

        with patch.object(DummyTypeChecker, "fit") as mock_fit, patch(
            "mlpype.base.pipeline.type_checker.logger"
        ) as mock_logger:

            type_checker.fit(dummy_data)
            mock_fit.assert_not_called()
            mock_logger.warning.assert_called_once_with("x has no supported type checker!")

    def test_transform(self, type_checker: DataSetTypeChecker, data: DataSet):
        type_checker.fit(data["x"], data["y"])

        with patch.object(type_checker.type_checkers["x"], "transform") as mock_x_transform, patch.object(
            type_checker.type_checkers["y"], "transform"
        ) as mock_y_transform:
            x, y = type_checker.transform(data["x"], data["y"])

            mock_x_transform.assert_called_once_with(data["x"])
            mock_y_transform.assert_called_once_with(data["y"])

            assert x == mock_x_transform.return_value
            assert y == mock_y_transform.return_value

    def test_fit_transform_integration(self, type_checker: DataSetTypeChecker, data: DataSet):
        type_checker.fit(data["x"], data["y"])

        # good
        trans_x, trans_y = type_checker.transform(data["x"], data["y"])
        assert trans_x == data["x"]
        assert trans_y == data["y"]

        # expected to fail
        with pytest_assert(AssertionError, "Provide a list!", exact=False):
            bad_data = {"z": 1}
            type_checker.transform(bad_data)

    def test_get_pydantic_types(self, type_checker: DataSetTypeChecker, data: DataSet):
        # done as integration test
        type_checker.fit(data["x"], data["y"])

        DummyDataSetModel: Type[DataSetModel] = type_checker.get_pydantic_types()

        model = DummyDataSetModel.to_model(data)

        assert model.x.data == data["x"]
        assert model.y.data == data["y"]
