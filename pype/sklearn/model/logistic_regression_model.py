from sklearn.linear_model import LogisticRegression

from pype.sklearn.model.sklearn_model import SklearnModel


class LogisticRegressionModel(SklearnModel):
    def __init__(
        self,
        inputs: list[str],
        outputs: list[str],
        model: LogisticRegression = None,
        penalty: str = "l2",
        dual: bool = False,
        tol: float = 1e-4,
        C: float = 1.0,
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
        class_weight: dict = None,
        solver: str = "lbfgs",
        max_iter: int = 100,
        multi_class: str = "auto",
        n_jobs: int = None,
        seed: int = 1,
    ) -> None:
        """Initialises a sklearn-LogisticRegression model ready for pype to use.

        Args:
            inputs (List[str]): A list of names of input Data. This determines which Data is
                used to fit the model.
            outputs (List[str]): A list of names of output Data. This determines the names of
                output Data.
            seed (int, optional): The RNG seed to ensure reproducability. Defaults to 1.
            model (LogisticRegression, optional): The LogisticRegression to use. Use `from_parameters` to initialise
                a model from just the parameters. Optional, if not set, we'll create a new model using the parameters.
            penalty (str, optional): See docs of `sklearn.linear.LogisticRegression`
            dual (bool, optional): See docs of `sklearn.linear.LogisticRegression`
            tol (float, optional): See docs of `sklearn.linear.LogisticRegression`
            C (float, optional): See docs of `sklearn.linear.LogisticRegression`
            fit_intercept (bool, optional): See docs of `sklearn.linear.LogisticRegression`
            intercept_scaling (float, optional): See docs of `sklearn.linear.LogisticRegression`
            class_weight (dict, optional): See docs of `sklearn.linear.LogisticRegression`
            solver (str, optional): See docs of `sklearn.linear.LogisticRegression`
            max_iter (int, optional): See docs of `sklearn.linear.LogisticRegression`
            multi_class (str, optional): See docs of `sklearn.linear.LogisticRegression`
            n_jobs (int, optional): See docs of `sklearn.linear.LogisticRegression`
        """
        if model is None:
            model = LogisticRegression(
                fit_intercept=fit_intercept,
                n_jobs=n_jobs,
                penalty=penalty,
                dual=dual,
                tol=tol,
                C=C,
                intercept_scaling=intercept_scaling,
                class_weight=class_weight,
                max_iter=max_iter,
                solver=solver,
                multi_class=multi_class,
            )
        super().__init__(inputs, outputs, model, seed=seed)
