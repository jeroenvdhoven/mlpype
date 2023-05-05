from unittest.mock import patch

from mlpype.sklearn.data import SqlSource


class TestSqlSource:
    def test_read(self):
        query = "Select * from *"
        conn = "sqlite://nowhere"
        extra = [4, 5]
        source = SqlSource(query, conn, 1, 2, extra=extra)

        with patch("mlpype.sklearn.data.sql_source.pd.read_sql") as mock_read_sql:
            result = source.read()

        mock_read_sql.assert_called_once_with(1, 2, sql=query, con=conn, extra=extra)
        assert result == mock_read_sql.return_value
