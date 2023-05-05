from typing import Any, Callable, Dict, Optional, Tuple, Union

from hyperopt.base import Trials
from hyperopt.fmin import fmin, space_eval
from hyperopt.mongoexp import MongoTrials

from mlpype.base.experiment import Experiment


def create_optimisation_function(
    experiment_template: Experiment,
    target_metric: Tuple[str, str],
    minimise_target: bool,
    seed: int,
) -> Callable[[dict], Union[float, int]]:
    """Create an optimisation function that hyperopt can use to find the best model.

    Args:
        experiment_template (Experiment): An example Experiment that will be run
            multiple times to find the best hyperparameters. We will create clean copies
            of this Experiment for every run.
        target_metric (Tuple[str, str]): The target metric to optimise. Should be of the form
            (dataset name, metric name). The dataset name needs to match one in the DataSet from the Experiment.
            The metric name should match a metric from the BaseEvaluator in the Experiment.
        minimise_target (bool): A boolean indicating if the target metric should be minimised or not.
        seed (int): The seed used for training.

    Returns:
        Callable[[dict], Union[float, int]]: The optimisation function to be used by hyperopt's fmin
    """

    def optimise(hyper_params: Dict[str, Any]) -> Union[float, int]:
        """Optimise a given Experiment at the give hyperparameters.

        Args:
            hyper_params (Dict[str, Any]): The hyperparameters to initialise
                a new Experiment with. Make sure these are properly prefixed with
                model__ and pipeline__<pipe name>__, as you would initialise a
                regular Experiment from arguments.

        Returns:
            Union[float, int]: The performance of this Experiment on the given parameters
                on one metric and dataset.
        """
        experiment = experiment_template.copy(hyper_params, seed=seed)
        metrics = experiment.run()
        dataset, metric = target_metric

        result = metrics[dataset][metric]
        assert isinstance(result, (float, int)), "The target metric should be a float or integer, please fix this."

        # Since we use fmin, we'll need to multiply our target with -1 if we want to maxisime our target instead.
        if minimise_target:
            return result
        else:
            return -result

    return optimise


def optimise_experiment(
    experiment_template: Experiment,
    search_space: Dict[str, Tuple[str, Callable]],
    target_metric: Tuple[str, str],
    max_evals: int,
    minimise_target: bool,
    trial_type: str = "normal",
    mongo_url: Optional[str] = None,
    mongo_exp_key: str = "experiment",
    trials: Optional[Trials] = None,
    training_seed: int = 1,
    **kwargs: Any,
) -> Tuple[Union[int, None], Union[float, None], Dict[str, Any], Trials]:
    """Optimise a mlpype Experiment given a search space using Hyperopt.

    We attempt to create a clean copy of Experiments between runs to preserve run independency.
    To help this, please ensure the input/output checkers can be refitted without issue, and
    the DataSources provide the same data even if called multiple times.

    Args:
        experiment_template (Experiment): An Experiment that will serve as the template for each run.
            We try to ensure a new, clean Experiment is created for each run. Please allow this by
            not sharing important, trained variables between Models or Operators of the same class.
            Each call to any DataSource provided should also return the same dataset.
        search_space (Dict[str, Tuple[str, Callable]]): A search space defined in the standard hyperopt way.
            We suggest providing this as a dictionary in the form:
                {<argument name>: (<name>, <hyperopt callable like hp.choice(...)>)}
            Make sure these are properly prefixed with model__ and pipeline__<pipe name>__, as you would
            initialise a regular Experiment from arguments.
        target_metric (Tuple[str, str]): The target metric to optimise. Should be of the form
            (dataset name, metric name). The dataset name needs to match one in the DataSet from the Experiment.
            The metric name should match a metric from the BaseEvaluator in the Experiment.
        max_evals (int): The maximum number of runs to perform for this hyperoptimisation search. Note that
            hyperopt will not create more runs if you provide a trials object with more trials than max_evals.
        minimise_target (bool): A boolean indicating if the target metric should be minimised or not.
        trial_type (str): If `trials` is not provided, we use this to create a new Trials object. There are currently 3
            options available:
                - 'normal': A regular Trials object, stored locally.
                - 'mongo': A MongoTrials object, storing trials in a mongodb. You will need to provide `mongo_url` and
                    `mongo_exp_key` to choose this type of trial.
                - 'spark': A SparkTrials object will be created. Currently not supported yet, WIP. Providing your own
                    SparkTrials object should still work.
        mongo_url (Optional[str]): The link to the mongodb that should be used.
            Only used if `trial_type` == 'mongo', and is required in that case.
        mongo_exp_key (str): The experiment name in the mongodb that should be used.
            Only used if `trial_type` == 'mongo'. We encourage you to set this to distinguish between runs.
            The default is "experiment"
        trials (Optional[Trials]): An existing Trials object with previous runs of the optimisation.
            Useful if you want to continue an old optimisation run without losing information.
            By default a new Trials object will be created.
        training_seed (int): Seed used to initialise a new Experiment (and run).
        **kwargs: Additional arguments to be provided to hyperopt's `fmin`.

    Returns:
        A tuple consisting of:
            - The best trial index (or None if no good trial was found)
            - The performance of the best trial (or None if no good trial was found)
            - The arguments for the best trial.
            - The Trials object created by this run.
    """
    # TODO: SparkTrials setup integration.
    optimise = create_optimisation_function(experiment_template, target_metric, minimise_target, seed=training_seed)

    if not minimise_target:
        print("Heads up: printed targets will be maximised. In print output, they will be multiplied with -1.")

    # initialise a trials object, if needed.
    if trials is None:
        if trial_type == "mongo":
            assert mongo_url is not None, "mongo_url can not be None if you want to run hyperopt with Mongo."
            assert mongo_url.endswith("/jobs"), "hyperopt currently required the DB name to end in /jobs."
            trials = MongoTrials(mongo_url, exp_key=mongo_exp_key)
            mongo_url_sub = "://".join(mongo_url.split("://")[1:])
            print(
                f"""
    Please make sure you start hyperopt mongo workers in a separate terminal!
    Currently, you can use `hyperopt-mongo-worker --mongo={mongo_url_sub} --poll-interval=0.1` If this fails,
    check https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB"""
            )
        elif trial_type == "spark":
            raise ValueError(
                "Spark trials have not been implemented yet with auto-setup."
                + "Provide a SparkTrials object to the trials argument instead."
            )
        else:
            trials = Trials()

    # Run the optimisation
    opt_result = fmin(optimise, search_space, max_evals=max_evals, trials=trials, **kwargs)
    best_arguments = space_eval(search_space, opt_result)

    # Get the best trial ID and corresponding performance.
    if trials.best_trial is None:
        best_trial_id = None
        performance = None
    else:
        best_trial_id = trials.best_trial["tid"]
        performance = trials.best_trial["result"]["loss"]
        if not minimise_target:
            performance = -performance

    return best_trial_id, performance, best_arguments, trials
