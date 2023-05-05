from tensorflow import Tensor  # type: ignore

from mlpype.base.data.data_source import DataSource


class TensorSource(DataSource[Tensor]):
    def __init__(self, tensor: Tensor) -> None:
        """A DataSource based around a created tensorflow Tensor.

        Good for testing purposes, but please use other sources for actual
        ML runs to prevent any data having to be hard-coded into the script.

        Args:
            tensor (Tensor): The Tensor to use as a source.
        """
        super().__init__()
        self.tensor = tensor

    def read(self) -> Tensor:
        """Returns the Tensor given as input before.

        Returns:
            Tensor: The Tensor used to initialise this object.
        """
        return self.tensor
