from pype.base.data.data_source import DataSource
from pype.base.data.dataset import DataSet
from pype.base.data.dataset_source import DataSetSource


class DummyDataSource(DataSource[int]):
    def __init__(self, v: int) -> None:
        super().__init__()
        self.v = v

    def read(self) -> int:
        return self.v


class Test_DataSetSource:
    def test_read(self):
        dss = DataSetSource[int](
            {
                "a": DummyDataSource(5),
                "b": DummyDataSource(10),
            }
        )

        result = dss.read()
        expected = {
            "a": 5,
            "b": 10,
        }
        assert isinstance(result, DataSet)
        assert len(expected) == len(result)
        assert expected["a"] == result["a"]
        assert expected["b"] == result["b"]
