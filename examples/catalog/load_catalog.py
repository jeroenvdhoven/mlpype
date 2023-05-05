"""This example shows how to use the DataCatalog from mlpype.

The catalog is located in `example_catalog.yml`

We show how to import data sources with basic parameters,
as well as more complex parameters.

We'll create a dummy table in Pyspark for this example to fetch data
from. The regular SQL query will not work.

Please run this example using:
python -m examples.catalog.load_catalog
"""
from pathlib import Path

import pandas as pd

# Generate data in pyspark for this example
from pyspark.sql import SparkSession

from mlpype.base.data import DataCatalog

s = SparkSession.builder.getOrCreate()
df = s.createDataFrame(pd.DataFrame({"column": [1, 2, 3, 4, 5]}))
df.createOrReplaceTempView("dummy_data")


path = Path(__file__).parent.absolute() / "example_catalog.yml"
print(f"Loading catalog from {path}")
assert path.is_file()

ctl = DataCatalog.from_yaml(path)

print(ctl)
