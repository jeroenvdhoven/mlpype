"""Provides tools for guaranteeing that SparkSession is active."""
from typing import Optional

from pyspark.sql import SparkSession


def guarantee_spark(
    spark_session: Optional[SparkSession] = None,
) -> SparkSession:
    """Checks if the given spark_session object is None and if so, returns the active session instead.

    Args:
        spark_session (Optional[SparkSession]): The spark_session object to guarantee. Defaults to None.

    Returns:
        SparkSession: The active SparkSession.

    Raises:
        AssertionError: If no SparkSession is active.
    """
    if spark_session is None:
        spark_session = SparkSession.getActiveSession()
    assert isinstance(spark_session, SparkSession), "No active SparkSession"
    return spark_session
