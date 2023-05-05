from unittest.mock import MagicMock, patch

from hyperopt import Trials, hp
from pytest import mark

from mlpype.base.experiment import Experiment
from mlpype.hyperopt.optimise import create_optimisation_function, optimise_experiment
from tests.shared_fixtures import dummy_experiment
from tests.utils import AnyArg, pytest_assert

dummy_experiment


class Test_create_optimisation_function:
    def test_create_optimisation_function(self):
        exp = MagicMock()
        exp_copy = MagicMock()
        exp.copy.return_value = exp_copy

        metrics = {"test": {"diff": 9, "other": 2}, "train": {"diff": 3}}
        exp_copy.run.return_value = metrics

        opt_func = create_optimisation_function(
            exp,
            ("test", "diff"),
            minimise_target=True,
            seed=1,
        )

        parameters_opt = {"model__a": 0}
        performance_opt = opt_func(parameters_opt)

        exp.copy.assert_called_once_with(parameters_opt, seed=1)
        exp_copy.run.assert_called_once_with()

        assert performance_opt == 9

    def test_create_optimisation_function_integration(self, dummy_experiment: Experiment):
        exp = dummy_experiment
        exp.model.a = 0
        baseline_performance = exp.run()

        opt_func = create_optimisation_function(
            exp,
            ("test", "diff"),
            minimise_target=True,
            seed=1,
        )

        parameters_opt = {"model__a": 0}
        performance_opt = opt_func(parameters_opt)

        assert performance_opt == baseline_performance["test"]["diff"]


class Test_optimise_experiment:
    def test_optimise_experiment(self):
        exp = MagicMock()
        exp_copy = MagicMock()
        exp.copy.return_value = exp_copy

        search_space = {
            "model__a": hp.randint("a", -10, 10),
        }
        metric = ("test", "diff")

        with patch("mlpype.hyperopt.optimise.create_optimisation_function") as mock_create, patch(
            "mlpype.hyperopt.optimise.fmin"
        ) as mock_fmin, patch("mlpype.hyperopt.optimise.space_eval") as mock_space, patch(
            "mlpype.hyperopt.optimise.Trials.best_trial"
        ) as mock_best_trial:
            best_trial_id, performance, best_arguments, trials = optimise_experiment(
                experiment_template=exp,
                search_space=search_space,
                target_metric=metric,
                max_evals=10,
                minimise_target=True,
                trial_type="normal",
                training_seed=1,
                fmin_arg=2390,
            )

            mock_create.assert_called_once_with(exp, metric, True, seed=1)
            mock_fmin.assert_called_once_with(
                mock_create.return_value, search_space, max_evals=10, trials=AnyArg(), fmin_arg=2390
            )
            mock_space.assert_called_once_with(search_space, mock_fmin.return_value)

            assert best_trial_id == mock_best_trial["tid"]
            assert performance == mock_best_trial["result"]["loss"]

            assert best_arguments == mock_space.return_value
            assert isinstance(trials, Trials)

    def test_trials_mongo(self):
        exp = MagicMock()
        exp_copy = MagicMock()
        exp.copy.return_value = exp_copy

        search_space = {
            "model__a": hp.randint("a", -10, 10),
        }
        metric = ("test", "diff")
        mongo_url = "mongo://db/jobs"
        exp_key = "experiment_key"

        with patch("mlpype.hyperopt.optimise.create_optimisation_function"), patch(
            "mlpype.hyperopt.optimise.fmin"
        ), patch("mlpype.hyperopt.optimise.space_eval"), patch("mlpype.hyperopt.optimise.MongoTrials") as mock_trial:
            optimise_experiment(
                experiment_template=exp,
                search_space=search_space,
                target_metric=metric,
                max_evals=10,
                minimise_target=True,
                trial_type="mongo",
                mongo_url="mongo://db/jobs",
                mongo_exp_key=exp_key,
                training_seed=1,
                fmin_arg=2390,
            )
            mock_trial.assert_called_once_with(mongo_url, exp_key=exp_key)

    @mark.parametrize(
        ["error", "url"],
        [
            ["mongo_url can not be None if you want to run hyperopt with Mongo.", None],
            ["hyperopt currently required the DB name to end in /jobs.", "something"],
        ],
    )
    def test_trials_mongo_asserts(self, url: str, error: str):
        exp = MagicMock()
        exp_copy = MagicMock()
        exp.copy.return_value = exp_copy

        search_space = {
            "model__a": hp.randint("a", -10, 10),
        }
        metric = ("test", "diff")
        mongo_url = "mongo://db/jobs"
        exp_key = "experiment_key"

        with patch("mlpype.hyperopt.optimise.create_optimisation_function"), patch(
            "mlpype.hyperopt.optimise.fmin"
        ), patch("mlpype.hyperopt.optimise.space_eval"), patch(
            "mlpype.hyperopt.optimise.MongoTrials"
        ) as mock_trial, pytest_assert(
            AssertionError, error
        ):
            optimise_experiment(
                experiment_template=exp,
                search_space=search_space,
                target_metric=metric,
                max_evals=10,
                minimise_target=True,
                trial_type="mongo",
                mongo_url=url,
                mongo_exp_key=exp_key,
                training_seed=1,
                fmin_arg=2390,
            )
            mock_trial.assert_called_once_with(mongo_url, exp_key=exp_key)

    def test_trials_normal(self):
        exp = MagicMock()
        exp_copy = MagicMock()
        exp.copy.return_value = exp_copy

        search_space = {
            "model__a": hp.randint("a", -10, 10),
        }
        metric = ("test", "diff")

        with patch("mlpype.hyperopt.optimise.create_optimisation_function"), patch(
            "mlpype.hyperopt.optimise.fmin"
        ), patch("mlpype.hyperopt.optimise.space_eval"), patch("mlpype.hyperopt.optimise.Trials") as mock_trial:
            optimise_experiment(
                experiment_template=exp,
                search_space=search_space,
                target_metric=metric,
                max_evals=10,
                minimise_target=True,
                trial_type="normal",
                training_seed=1,
                fmin_arg=2390,
            )
            mock_trial.assert_called_once_with()

    def test_trials_spark(self):
        exp = MagicMock()
        exp_copy = MagicMock()
        exp.copy.return_value = exp_copy

        search_space = {
            "model__a": hp.randint("a", -10, 10),
        }
        metric = ("test", "diff")

        with patch("mlpype.hyperopt.optimise.create_optimisation_function"), patch(
            "mlpype.hyperopt.optimise.fmin"
        ), patch("mlpype.hyperopt.optimise.space_eval"), pytest_assert(
            ValueError,
            "Spark trials have not been implemented yet with auto-setup."
            + "Provide a SparkTrials object to the trials argument instead.",
        ):
            optimise_experiment(
                experiment_template=exp,
                search_space=search_space,
                target_metric=metric,
                max_evals=10,
                minimise_target=True,
                trial_type="spark",
                training_seed=1,
                fmin_arg=2390,
            )

    def test_trials_provided(self):
        exp = MagicMock()
        exp_copy = MagicMock()
        exp.copy.return_value = exp_copy

        search_space = {
            "model__a": hp.randint("a", -10, 10),
        }
        metric = ("test", "diff")
        trials = MagicMock()

        with patch("mlpype.hyperopt.optimise.create_optimisation_function") as mock_create, patch(
            "mlpype.hyperopt.optimise.fmin"
        ) as mock_fmin, patch("mlpype.hyperopt.optimise.space_eval"):
            optimise_experiment(
                experiment_template=exp,
                search_space=search_space,
                target_metric=metric,
                max_evals=10,
                minimise_target=True,
                trial_type="normal",
                training_seed=1,
                trials=trials,
                fmin_arg=2390,
            )
            mock_fmin.assert_called_once_with(
                mock_create.return_value, search_space, max_evals=10, trials=trials, fmin_arg=2390
            )

    def test_maxisime_target_result(self):
        pass

    def test_optimise_experiment_integration(self, dummy_experiment: Experiment):
        exp = dummy_experiment
        # prevent logging
        exp.experiment_logger = MagicMock()
        search_space = {
            "model__a": hp.randint("a", -10, 10),
        }
        metric = ("test", "diff")
        n_evals = 6

        best_trial_id, performance, best_arguments, trials = optimise_experiment(
            experiment_template=exp,
            search_space=search_space,
            target_metric=metric,
            max_evals=n_evals,
            minimise_target=True,
            trial_type="normal",
            training_seed=1,
        )

        assert best_trial_id is not None
        assert n_evals >= best_trial_id and best_trial_id >= 0
        assert trials is not None
        assert performance == trials.best_trial["result"]["loss"]

        # Higher a leads to better performance. Highest a should have been selected.
        max_a = max([t["misc"]["vals"]["a"] for t in trials.trials])
        assert max_a == best_arguments["model__a"]
