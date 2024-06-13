from unittest.mock import MagicMock, patch

from mlpype.spark.data import SparkSqlSource


class Test_SparkSqlSource:
    def test(self):
        session = MagicMock()
        query = "select * from table"

        with patch("mlpype.spark.data.spark_sql_source.guarantee_spark") as mock_guarantee:
            source = SparkSqlSource(query, session)

        mock_guarantee.assert_called_once_with(session)
        result = source.read()

        mock_session = mock_guarantee.return_value
        mock_session.sql.assert_called_once_with(query)
        assert result == mock_session.sql.return_value

    def test_without_session(self):
        query = "select * from table"

        with patch("mlpype.spark.data.spark_sql_source.guarantee_spark") as mock_guarantee:
            source = SparkSqlSource(query)
            result = source.read()

        mock_guarantee.assert_called_once_with(None)
        session = mock_guarantee.return_value

        session.sql.assert_called_once_with(query)
        assert result == session.sql.return_value
