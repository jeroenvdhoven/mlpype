from sklearn.linear_model import LinearRegression

from pype.sklearn.model.sklearn_model import SklearnModel


class LinearRegressionModel(SklearnModel):
    def __init__(
        self,
        inputs: list[str],
        outputs: list[str],
        model: LinearRegression,
        seed: int = 1,
    ) -> None:
        """Initialises a sklearn-LinearRegression model ready for pype to use.

        Args:
            inputs (List[str]): A list of names of input Data. This determines which Data is
                used to fit the model.
            outputs (List[str]): A list of names of output Data. This determines the names of
                output Data.
            seed (int, optional): The RNG seed to ensure reproducability. Defaults to 1.
            model (LinearRegression): The LinearRegression to use. Use `from_parameters` to initialise
                a model from just the parameters.
        """
        super().__init__(inputs, outputs, model, seed=seed)

    @classmethod
    def from_parameters(
        cls,
        inputs: list[str],
        outputs: list[str],
        seed: int = 1,
        fit_intercept: bool = True,
        normalize: str = "deprecated",
        copy_X: bool = True,
        n_jobs: int = None,
        positive: bool = False,
    ) -> "LinearRegressionModel":
        """Initialises a sklearn-LinearRegression model ready for pype to use from parameters.

        Args:
            inputs (List[str]): A list of names of input Data. This determines which Data is
                used to fit the model.
            outputs (List[str]): A list of names of output Data. This determines the names of
                output Data.
            seed (int, optional): The RNG seed to ensure reproducability. Defaults to 1.
            fit_intercept (bool, optional): See docs of `sklearn.linear.LinearRegression`
            normalize (str, optional): See docs of `sklearn.linear.LinearRegression`
            copy_X (bool, optional): See docs of `sklearn.linear.LinearRegression`
            n_jobs (int, optional): See docs of `sklearn.linear.LinearRegression`
            positive (bool, optional): See docs of `sklearn.linear.LinearRegression`

        Returns:
            LinearRegressionModel: A new pype-compliant LinearRegression Model.
        """
        model = LinearRegression(
            fit_intercept=fit_intercept,
            normalize=normalize,
            copy_X=copy_X,
            n_jobs=n_jobs,
            positive=positive,
        )

        return cls(inputs, outputs, model, seed=seed)
