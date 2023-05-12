from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from pytest import mark

from mlpype.base.data.data_catalog import DataCatalog
from mlpype.base.data.data_source import DataSource
from mlpype.base.data.dataset import DataSet
from tests.utils import pytest_assert


class DummyDataSource(DataSource[int]):
    def __init__(self, v: int) -> None:
        super().__init__()
        self.v = v

    def read(self) -> int:
        return self.v

    @staticmethod
    def dummy_method(p):
        return p + 5


class Test_DataCatalog:
    def test_read(self):
        dss = DataCatalog[int](
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

    @mark.parametrize(
        ["expectation", "dictionary"],
        [
            [False, {}],
            [False, {"1": 1, "2": 2, "3": 3}],
            [False, {"callable": "", "-": {}}],
            [False, {"-": "", "args": {}}],
            [False, {"-": "", "args": []}],
            [True, {"callable": "", "args": {}}],
        ],
    )
    def test_is_valid_parseable_object(self, expectation: bool, dictionary: dict):
        result = DataCatalog._is_valid_parseable_object(dictionary)
        assert result == expectation

    def test_load_class(self):
        path = "mlpype.base.data.DataSource"
        result = DataCatalog._load_class(path)

        assert result == DataSource

    def test_load_class_assert(self):
        path = "mlpype.base.data.DataSource:read:failure"

        with pytest_assert(AssertionError, "We do not accept paths with more than 1 `:`"):
            DataCatalog._load_class(path)

    def test_load_class_with_method(self):
        path = "mlpype.base.data.DataSource:read"
        result = DataCatalog._load_class(path)

        assert result == DataSource.read

    def test_parse_object(self):
        v = 10
        dct = {"callable": "tests.base.data.test_data_catalog.DummyDataSource", "args": {"v": v}}
        result = DataCatalog._parse_object(dct)

        assert isinstance(result, DummyDataSource)
        assert result.v == v

    def test_nested_parse_object(self):
        p = 2
        dct = {"callable": "tests.base.data.test_data_catalog.DummyDataSource:dummy_method", "args": {"p": p}}
        result = DataCatalog._parse_object(dct)

        assert result == 7

    def test_from_yaml(self):
        path = Path(__file__).parent / "config.yml"

        catalog = DataCatalog.from_yaml(path)

        assert len(catalog) == 2
        assert "dataframe" in catalog
        assert "pandas_sql" in catalog

        # pandas_sql
        sql_source = catalog["pandas_sql"]
        assert sql_source.sql == "select * from database.table"
        assert sql_source.con == "http://<your database url>"

        # dataframe
        expected_df = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": ["a", "b"],
            }
        )
        assert_frame_equal(expected_df, catalog["dataframe"].read())

    def test_from_yaml_with_jinja(self):
        path = Path(__file__).parent / "config_with_jinja.yml"

        params = {
            "pandas_sql": {"sql": "select * from database.table", "con": "http://<your database url>"},
            "dataframe": {"x": 4.0},
        }
        catalog = DataCatalog.from_yaml(path, parameters=params)

        assert len(catalog) == 2
        assert "dataframe" in catalog
        assert "pandas_sql" in catalog

        # pandas_sql
        sql_source = catalog["pandas_sql"]
        assert sql_source.sql == "select * from database.table"
        assert sql_source.con == "http://<your database url>"

        # dataframe
        expected_df = pd.DataFrame(
            {
                "x": [1.0, 4.0],
                "y": ["a", "b"],
            }
        )
        assert_frame_equal(expected_df, catalog["dataframe"].read())
