"""This module contains the base data types for mlpype.

We include:

- DataSource: An interface for reading data. The main way to define used datasets in MLpype.
- DataSink: An interface for writing data.
- DataCatalog: A container for DataSources. This allows you to keep track of all your data sources in one place, and \
    load them all in one go.
- DataSet: A collection of read data. This is used internally to pass your data between pipelines and models.

"""
from .data_catalog import DataCatalog
from .data_sink import DataSink
from .data_source import DataSource
from .dataset import DataSet
