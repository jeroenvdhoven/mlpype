from sklearn.preprocessing import StandardScaler

from mlpype.sklearn.pipeline.numpy_type_checker import NumpyTypeChecker
from mlpype.sklearn.pipeline.pandas_type_checker import PandasTypeChecker

tcc = [NumpyTypeChecker, PandasTypeChecker]


class CustomStandardScaler(StandardScaler):
    pass
