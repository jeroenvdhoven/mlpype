from pype.base.data import DataSet, DataSetSource
from pype.base.model import Model
from pype.base.pipeline import Pipeline


class Inferencer:
    def __init__(self, model: Model, pipeline: Pipeline):
        """Provides a standard way of inferencing with pype models.

        Args:
            model (Model): The Model to use in inference.
            pipeline (Pipeline): The Pipeline to use in inference.
        """
        self.model = model
        self.pipeline = pipeline

    def predict(self, data: DataSet | DataSetSource) -> DataSet:
        """Predicts using the given data using the Pipeline and Model.

        # TODO: inverse transformation.

        Args:
            data (DataSet | DataSetSource): The data to predict for.

        Returns:
            DataSet: The predictions from Model.
        """
        if isinstance(data, DataSetSource):
            data = data.read()

        transformed = self.pipeline.transform(data)
        predicted = self.model.transform(transformed)
        return predicted
