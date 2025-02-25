"""
The MLPype base package provides all the core components to run a one stop shop ML experiment.

Most of these are configured
as interfaces, allowing the library to be easily extended.

Core classes to look into are:

- The Experiment class. The main functionality resides in `experiment.Experiment`. \
    The Experiment class will combine the other components of this library and run a standardised ML experiment.
- The DataCatalog. Experiments use DataCatalogs to load data. This allows you to define your data as config, \
    so it doesn't immediately need to be loaded into memory.
- The Pipeline and Pipe classes. These are very similar to the sklearn Pipeline, with the exception that they handle \
    data inputs in dictionary format. This gives much more freedom when transforming and combining datasets in the \
        preprocessing step.
- The Evaluator class. This is a standardised way of evaluating models. The BaseEvaluator uses a dictionary of \
    {name: function} to evaluate a model.

Core interfaces to look into are:

- The Model class. This is the base class for all models. The setup is such that it should allow you to easily \
    integrate other packages into your model. We have already integrated the following packages:
    - sklearn
    - tensorflow / keras
    - pyspark
    - xgboost / lightgbm
- The DataSource class. This is the base class for all data sources. This allows easy extension of inputs to multiple \
    data sources, ranging from flat files to databases, or any other source.
"""
from . import data, deploy, evaluate, experiment, logger, model, pipeline, serialiser, utils
