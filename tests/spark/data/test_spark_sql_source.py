from unittest.mock import MagicMock

from pype.spark.data import SparkSqlSource


class Test_SparkSqlSource:
    def test(self):
        session = MagicMock()
        query = "select * from table"

        source = SparkSqlSource(session, query)
        result = source.read()

        session.sql.assert_called_once_with(query)
        assert result == session.sql.return_value
