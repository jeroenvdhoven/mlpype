"""Provides wrapper classes for serialising and deserialising Spark components."""
from pyspark.ml import Model as BaseSparkModel
from pyspark.ml import Predictor, Transformer
from pyspark.ml.util import MLReader, MLWritable


class SerialisablePredictor(Predictor, MLWritable, MLReader):
    """A Spark Predictor that can be serialised and deserialised."""


class SerialisableTransformer(Transformer, MLWritable, MLReader):
    """A Spark Transformer that can be serialised and deserialised."""


class SerialisableSparkModel(BaseSparkModel, MLWritable, MLReader):
    """A Spark Model that can be serialised and deserialised."""
