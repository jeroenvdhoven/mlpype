from pytest import fixture

from mlpype.base.data.dataset import DataSet


class Test_DataSet:
    @fixture
    def data(self):
        return DataSet[int](
            {
                "a": 2,
                "b": 4,
            }
        )

    def test_len(self, data: DataSet[int]):
        assert len(data) == 2

    def test_get_item(self, data: DataSet[int]):
        assert data["a"] == 2

    def test_get_all(self, data: DataSet[int]):
        result = data.get_all(["b", "a"])

        assert result == [4, 2]

    def test_copy(self, data: DataSet[int]):
        res = data.copy()

        assert len(res) == len(data)
        assert res["a"] == data["a"]
        assert res["b"] == data["b"]

    def test_set_item(self, data: DataSet[int]):
        data["c"] = 9
        assert "c" in data
        assert data["c"] == 9

    def test_set_all(self, data: DataSet[int]):
        data.set_all(["c", "d"], [8, 9])

        assert "c" in data
        assert data["c"] == 8
        assert "d" in data
        assert data["d"] == 9

    def test_contains(self, data: DataSet[int]):
        assert "a" in data
        assert "c" not in data
