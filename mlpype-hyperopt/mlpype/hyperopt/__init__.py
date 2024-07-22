"""An integration for hyperopt with MLpype.

This package aims to use the Experiment setup of MLpype to perform hyperparameter optimisation.

We expose 1 function, `optimise_experiment`, which can be used to perform hyperparameter \
    optimisation given an Experiment.
"""
from .optimise import optimise_experiment
