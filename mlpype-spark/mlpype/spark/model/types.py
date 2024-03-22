from pyspark.ml import Model as BaseSparkModel
from pyspark.ml import Predictor, Transformer
from pyspark.ml.util import MLReader, MLWritable


class SerialisablePredictor(Predictor, MLWritable, MLReader):
    pass


class SerialisableTransformer(Transformer, MLWritable, MLReader):
    pass


class SerialisableSparkModel(BaseSparkModel, MLWritable, MLReader):
    pass
