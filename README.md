# Pype: standardise model training across libraries

- short description of `why`?
- Usage and examples
- Modular
- features
    - Training
    - Loading
    - Deploying
- Support
- Contribution
    - how to setup `dev`
    - standards


# Features

## Model training

The goal is to help standardise the following steps:
- Fetching data
- Pipeline fitting
- Model training
- Serialisation
- Logging

We aim to do this by providing basic interfaces that our code works with that should be easy to implement for other packages.
This should help keep the base package lightweight but flexible.

## Model loading
With most steps standardised, it becomes a lot easier to standardise model deployment as well.

## Subpackages
Pype was setup with the intent of being modular. The base package provides the interfaces (e.g. `Model`, `DataSource` classes) and interactions between (e.g. `Experiment` class).


Currently the following subpackages are available:

- `pype.base`: The base package. Required for other packages. Provides basics of training Pype-compliant models.
    - Fetching data
    - Pipeline fitting
    - Model training
    - Serialisation
    - Logging
    - Deployment
- `pype.mlflow`: MLflow integration package. Allows easy use of logging through mlflow and downloading/loading models trained through Pype and logged on MLflow.
- `pype.sklearn`: Sklearn integration package. Simplifies training sklearn models on Pype.
- `pype.xgboost`: Minor extension to the `pype.sklearn` package, including XGBoost models.
- `pype.tensorflow`: Incorporates tensorflow/keras into Pype.
- `pype.hyperopt`: Use hyperopt to do hyperparameter tuning in Pype.

# Contributing

## Basics

## Setting up development environment

## Coding standards

## The ToDo list:
- `spark` integration
- `pytorch` integration
- split fastapi endpoint out from `base` package