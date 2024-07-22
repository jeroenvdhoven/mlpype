"""Creates a default spark session."""
from pyspark.sql import SparkSession


def get_default_session() -> SparkSession:
    """Creates a default spark session.

    Returns:
        SparkSession: A default spark session.
    """
    return SparkSession.builder.getOrCreate()
