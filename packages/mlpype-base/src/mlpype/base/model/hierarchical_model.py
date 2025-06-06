"""Provides a hierarchical Model class, capable of nesting multiple models underneath."""
import types
import typing
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Tuple, Type, TypeVar, Union

from loguru import logger

from mlpype.base.model.model import Model
from mlpype.base.serialiser.joblib_serialiser import JoblibSerialiser

T = TypeVar("T", bound=Model)
Data = TypeVar("Data")


class HierarchicalModel(Model, Generic[T]):
    """The base class for hierarchical models.

    A hierarchical model splits the data into multiple groups.

    Each dataset provided to the model needs to be splittable using the data_splitter function.
    Data is automatically recombined after prediction using the data_merger function.
    """

    HIERARCHICAL_MODEL_FOLDER = "hierarchical_model"
    HIERARCHICAL_MODEL_SUBFOLDER = "models"
    SPLITTER_FILENAME = "data_splitter.pkl"
    MERGER_FILENAME = "data_merger.pkl"

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        data_splitter: Callable[[Data], Dict[str, Data]],
        data_merger: Callable[[Dict[str, Data]], Data],
        seed: int = 1,
        model: Optional[Dict[str, T]] = None,
        **model_args: Any,
    ) -> None:
        """The base class for hierarchical models.

        A hierarchical model splits the data into multiple groups.

        Each dataset provided to the model needs to be splittable using the data_splitter function.
        Data is automatically recombined after prediction using the data_merger function.

        Each submodel will be created on the fly during training. You can provide `model` if you
        already have a trained dictionary.

        Below are some examples for how to do this yourself.

        ```python
        # Works
        from mlpype.sklearn.model import LinearRegressionModel
        class HierarchicalLRModel(HierarchicalModel[LinearRegressionModel]):
            pass

        # An alternative to dynamically generate the model, which is easier to export/import
        model_class = HierarchicalModel.class_from_model_class(LinearRegressionModel)

        # Unfortunately, using something like the following will not work due to how Generic types are handled.
        HierarchicalLRModel = HierarchicalModel[LinearRegressionModel]
        ```

        Args:
            inputs (List[str]): A list of names of input Data. This determines which Data is
                used to fit the model.
            outputs (List[str]): A list of names of output Data. This determines the names of
                output variables.
            data_splitter (Callable[[Data], Dict[str, Data]]): A function to split incoming Data into
                keyed slices.
            data_merger (Callable[[Dict[str, Data]], Data]): A function to merge keyed Data slices into
                a single Data object.
            seed (int, optional): The RNG seed to ensure reproducability.. Defaults to 1.
            model (Optional[Dict[str, T]], optional): Optional dictionary of pretrained submodels.
                If not set, an empty list will be used (and the model needs to be trained first).
            **model_args (Any): Keyword arguments that will be passed to each `model` to initialise them
                during training. Besides these, `inputs`, `outputs`, and `seed` will be send by default.
        """
        if model is None:
            model = {}
        self.model = model

        super().__init__(inputs, outputs, seed)

        self.data_splitter = data_splitter
        self.data_merger = data_merger
        self.model_args = model_args

    @classmethod
    def _get_annotated_class(cls) -> Type[T]:
        return typing.get_args(cls.__orig_bases__[0])[0]

    def _split_data(self, *data: Data) -> Dict[str, List[Data]]:
        logger.info("Splitting dataset")
        split_data = defaultdict(list)
        for d in data:
            split = self.data_splitter(d)
            for k, v in split.items():
                split_data[k].append(v)

        number_of_different_dataset = {len(v) for v in split_data.values()}
        assert len(number_of_different_dataset) == 1, f"Data is not split correctly, got {number_of_different_dataset}"
        return split_data

    def _merge_data(self, data: Dict[str, Union[Tuple[Data], Data]]) -> List[Data]:
        logger.info("Merging dataset")

        data_list: List[Dict[str, Data]] = []
        for key, d in data.items():
            if len(self.outputs) == 1:
                d = (d,)  # type: ignore
            assert isinstance(d, tuple)

            if len(data_list) == 0:
                data_list = [{} for _ in range(len(d))]

            for i, subset in enumerate(d):
                data_list[i][key] = subset

        return [self.data_merger(subdict) for subdict in data_list]

    def _fit(self, *data: Any) -> None:
        split_data = self._split_data(*data)

        self.model = {}
        logger.info("Fitting model")
        for key, dataset in split_data.items():
            logger.info(f"Fitting for key: {key}")
            model = self._get_annotated_class()(
                inputs=self.inputs,
                outputs=self.outputs,
                seed=self.seed,
                **self.model_args,
            )
            model._fit(*dataset)
            self.model[key] = model

    def _transform(self, *data: Data) -> Union[Iterable[Data], Data]:
        split_data = self._split_data(*data)

        result = {}
        for key, dataset in split_data.items():
            logger.info(f"Transforming for key: {key}")
            if key not in self.model:
                logger.error(f"Model for key {key} not found. Please fit model first!")
            model = self.model[key]
            result[key] = model._transform(*dataset)

        res = self._merge_data(result)
        if len(self.outputs) == 1:
            # If we only return 1 output, the `transform` functions expects no tuple
            return res[0]
        return res

    def _save(self, folder: Path) -> None:
        model_folder = folder / self.HIERARCHICAL_MODEL_FOLDER
        model_folder.mkdir(exist_ok=True)

        joblib_serialiser = JoblibSerialiser()
        joblib_serialiser.serialise(self.data_splitter, model_folder / self.SPLITTER_FILENAME)
        joblib_serialiser.serialise(self.data_merger, model_folder / self.MERGER_FILENAME)

        model_subfolder = model_folder / self.HIERARCHICAL_MODEL_SUBFOLDER
        model_subfolder.mkdir(exist_ok=True)
        for name, model in self.model.items():
            model.save(model_subfolder / name)

    @classmethod
    def _load(
        cls: Type["HierarchicalModel"],
        folder: Path,
        inputs: List[str],
        outputs: List[str],
    ) -> "HierarchicalModel":
        model_subclass = cls._get_annotated_class()
        model_folder = folder / cls.HIERARCHICAL_MODEL_FOLDER
        assert model_folder.is_dir(), f"Folder {model_folder} does not exist"

        joblib_serialiser = JoblibSerialiser()
        data_splitter = joblib_serialiser.deserialise(model_folder / cls.SPLITTER_FILENAME)
        data_merger = joblib_serialiser.deserialise(model_folder / cls.MERGER_FILENAME)

        model_subfolder = model_folder / cls.HIERARCHICAL_MODEL_SUBFOLDER
        assert model_subfolder.is_dir(), f"Folder {model_subfolder} does not exist"

        models = {model_name.name: model_subclass.load(model_name) for model_name in model_subfolder.iterdir()}

        return cls(
            inputs=inputs,
            outputs=outputs,
            data_splitter=data_splitter,
            data_merger=data_merger,
            model=models,
        )

    def set_seed(self) -> None:
        """Sets the seed for all submodels.

        Since no models are created at creation time, this will not work. Instead, set_seed is called
        automatically when the models are created at training time.

        This function will reset each seed.
        """
        for model in self.model.values():
            model.set_seed()

    @classmethod
    def class_from_model_class(
        cls,
        model_class: Type[Model],
    ) -> Type["HierarchicalModel"]:
        """Create a HierarchicalModel wrapper class from a Model class.

        This should support all MLpype compliant models.

        Args:
            model_class (Type[Model]): The class of the MLpype submodel. For example,
                LinearRegressionModel or LogisticRegressionModel from the sklearn subpackage.

        Returns:
            Type[HierarchicalModel]: The created HierarchicalModel.
        """
        new_name = f"Hierarchical{model_class.__name__}"
        try:
            old_docs = cls.__doc__
            assert isinstance(old_docs, str)
        except AttributeError:
            logger.warning(f"Failed to add docstring to {new_name}.")
            old_docs = """The base class for hierarchical models.

    A hierarchical model splits the data into multiple groups.

    Each dataset provided to the model needs to be splittable using the data_splitter function.
    Data is automatically recombined after prediction using the data_merger function.
    """

        new_doc = (
            old_docs
            + f"""
See HierarchicalModel for the original source and docs for subfunctions.

This model uses a fixed class: {model_class.__name__}.
The source module is: {model_class.__module__}
"""
        )

        klass = types.new_class(
            new_name,
            (HierarchicalModel[model_class],),  # type: ignore
            exec_body=lambda ns: ns.update(
                {
                    "__doc__": new_doc,
                    "__module__": HierarchicalModel.__module__,
                }
            ),
        )

        return klass

    @classmethod
    def from_model_class(
        cls,
        model_class: Type[Model],
        inputs: List[str],
        outputs: List[str],
        data_splitter: Callable[[Data], Dict[str, Data]],
        data_merger: Callable[[Dict[str, Data]], Data],
        seed: int = 1,
        **model_args: Any,
    ) -> "HierarchicalModel":
        """Create a HierarchicalModel from a Model class.

        This should support all MLpype compliant models.

        Args:
            model_class (Type[Model]): The class of the MLpype submodel. For example,
                LinearRegressionModel or LogisticRegressionModel from the sklearn subpackage.
            inputs (List[str]): A list of names of input Data. This determines which Data is
                used to fit the model.
            outputs (List[str]): A list of names of output Data. This determines the names of
                output variables.
            data_splitter (Callable[[Data], Dict[str, Data]]): A function to split incoming Data into
                keyed slices.
            data_merger (Callable[[Dict[str, Data]], Data]): A function to merge keyed Data slices into
                a single Data object.
            seed (int, optional): The RNG seed to ensure reproducability.. Defaults to 1.
            **model_args (Any): Keyword arguments that will be passed to each `model` to initialise them
                during training. Besides these, `inputs`, `outputs`, and `seed` will be send by default.

        Returns:
            HierarchicalModel: The created HierarchicalModel.
        """
        new_cls = cls.class_from_model_class(model_class)
        return new_cls(
            inputs=inputs,
            outputs=outputs,
            seed=seed,
            data_splitter=data_splitter,
            data_merger=data_merger,
            model=None,
            **model_args,
        )
