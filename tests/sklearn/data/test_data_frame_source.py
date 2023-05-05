import pandas as pd
from pandas.testing import assert_frame_equal

from mlpype.sklearn.data.data_frame_source import DataFrameSource


def test_DataFrameSource():
    df = pd.DataFrame({"x": [1, 2, 3]})

    source = DataFrameSource(df.copy())

    assert_frame_equal(df, source.read())
