from typing import Any

import pandas as pd

from mlpype.base.data.data_source import DataSource


class SqlSource(DataSource[pd.DataFrame]):
    def __init__(
        self,
        sql: str,
        con: str,
        *args: Any,
        **kwargs: Any,
    ):
        """Reads data from a sql table using Pandas read_sql.

        This class only works as a storage for the arguments to call Pandas read_sql.
        For proper documentation on the arguments, check Pandas read_sql.

        Args:
            sql (str): The sql and the `sql` argument to `Pandas read_sql`.
            con (str): The con string and `con` argument to `Pandas read_sql`.
            args: Positional arguments to be passed to `Pandas read_sql`.
            kwargs: Keyword arguments to be passed to `Pandas read_sql`.
        """
        self.sql = sql
        self.con = con
        self.args = args
        self.kwargs = kwargs
        super().__init__()

    def read(self) -> pd.DataFrame:
        """Use the provided arguments to call `read_sql`.

        Returns:
            pd.DataFrame: The result of the provided query.
        """
        return pd.read_sql(*self.args, sql=self.sql, con=self.con, **self.kwargs)
