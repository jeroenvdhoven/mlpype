from sklearn.preprocessing import StandardScaler

from pype.sklearn.pipeline.numpy_type_checker import NumpyTypeChecker
from pype.sklearn.pipeline.pandas_type_checker import PandasTypeChecker

tcc = [NumpyTypeChecker, PandasTypeChecker]


class CustomStandardScaler(StandardScaler):
    pass
