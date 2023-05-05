# MLpype: standardise model training across libraries

## Why yet another ML package?

## Usage and examples
There are a couple of examples available of training scripts using MLpype and various subpackages in the `examples` folder.

In general, you'll need the following components:
- DataCatalog's: a collection of DataSource's, that can be read to produce a DataSet. This will specify which data is used by your
model's input and how to collect it. This can range from cloud-storage files, SQL databases, etc.
- Pipeline: Another trainable pipeline format? Yes. Those familiar with kedro will see some similarities, but we specify a difference
between the fitting and transforming phase, much like sklearn's pipelines. The framework should be flexible enough to allow a variety
of packages to integrate with the base Operator class, or at least comply with its interface.
- Model: An interface that should detail well what functions are needed to make a Model work. We have intergrations and examples for sklearn, pyspark,
and keras models, but you can also make your own integration.
- Evaluator:
- ExperimentLogger: 
- input/output type checkers:

Optionally, you can specify the following:
- serialiser
- output_folder
- additional_files_to_store

You can check the documentation for `Experiment` for more details.

## Advantages
- Modular: `MLpype` is setup as a namespace package. This allows you to install parts of the library to suit your needs.
    Development is done by default on a full installation of `MLpype`, so all packages *should* work together.
- Standardised: Don't code the same old steps every single time, just re-use them. `MLpype` handles a lot of good standard
    steps in ML training, like experiment logging and keeping track of artifacts.
- Extensive: Currently `MLpype` has a couple of integrations with other popular packages like `sklearn` and `tensorflow`, but
    also `mlflow`. Ideally we'd like to expand this to other common ML libraries and also platforms. We'd like to see this
    become a simple, universal starting point for machine learning with proper standardisation
- Flexibility: Bring your own models, data sources, and preprocessing steps. Bring your own loggers and serialisers. Just
    follow the interfaces we've set up, and you should be good to go!

# Features
## Model training

The goal is to help standardise the following steps:
- Fetching data
- Pipeline fitting
- Model training
- Serialisation
- Logging

We aim to do this by providing basic interfaces that should be easy to implement for other packages.
This should help keep the base package lightweight but flexible, while allowing subpackages to implement
specific functionality for other packages. You can also directly extend the base interfaces to work with
`MLpype`.

## Model loading
With most steps standardised, it becomes a lot easier to standardise model deployment as well. Currently we offer:

- Loading your model in a standardised way back into memory from an output folder
- Deploying your model using a FastAPI server
- Packaging your model as a wheel package, with dependencies linked from the experiment.
    - It is possible to extend this wheel package with your own functions as well, such as the FastAPI server

## Subpackages
MLpype is setup in a modular way. The base package provides the interfaces (e.g. `Model`, `DataSource` classes) and interactions between them (e.g. `Experiment` class).
Its dependency footprint is quite low to make sure no excess packages are installed when working with `MLpype` in different ways.

Subpackages like `MLpype.sklearn` provide implementations, such as sklearn, tensorflow, or mlflow integrations. Not all subpackages need to be installed,
making deployments lighter and less prone to dependency problems. This also reduces dependencies between subpackages, reducing complicated internal dependency issues.


Currently the following subpackages are available:

- `MLpype.base`: The base package. Required for other packages. Provides basics of training MLpype-compliant models.
    - Fetching data
    - Pipeline fitting
    - Model training
    - Serialisation
    - Logging
    - Deployment
- `MLpype.mlflow`: MLflow integration package. Allows easy use of logging through mlflow and downloading/loading models trained through MLpype and logged on MLflow.
- `MLpype.sklearn`: Sklearn integration package. Simplifies training sklearn models on MLpype.
- `MLpype.xgboost`: Minor extension to the `MLpype.sklearn` package, including XGBoost models.
- `MLpype.tensorflow`: Incorporates tensorflow/keras into MLpype.
- `MLpype.hyperopt`: Use hyperopt to do hyperparameter tuning in MLpype.
- `MLpype.spark`: In progress, will provide pyspark integrations for MLpype.

# Contributing

## Basics

Besides regular good coding practices, we'd like to ask you to use type hints where possible and realistic.
This greatly helps the users and other developers develop using your new code quicker. We have mypy enabled
to help you out and find missing type hints quicker.
## Setting up development environment

If you're interested in helping out, you can install all packages into your environment using `make dev-install`.
Please make sure to also install the pre-commit steps using `make pre-commit-install`. Check out the `.pre-commit-config.yaml`
file for the exact steps we use, but it's summarised as:

- sorting imports
- remove unused imports
- black
- mypy for type hitns
