from typing import Any, Dict, List, Optional, Union

from mlpype.base.data import DataSet
from mlpype.base.pipeline.pipe import Pipe
from mlpype.base.utils.parsing import get_args_for_prefix


class Pipeline:
    def __init__(self, pipes: List[Union[Pipe, "Pipeline"]]) -> None:
        """A pipeline of operations that can be re-applied to new, similar Data.

        Args:
            pipes (List[Union[Pipe, Pipeline]]): A list of either Pipes
                or other Pipelines. These form the steps that should be applied.
        """
        self.pipes = pipes
        self._assert_all_names_different()

    def _assert_all_names_different(self, names: Optional[set] = None) -> None:
        if names is None:
            names = set()

        for pipe in self.pipes:
            if isinstance(pipe, Pipeline):
                pipe._assert_all_names_different(names)
            else:
                assert pipe.name not in names, f"{pipe.name} is used multiple times."
                names.add(pipe.name)

    def fit(self, data: DataSet) -> "Pipeline":
        """Fits the entire Pipeline to the given DataSet.

        Args:
            data (DataSet): The DataSet to use in fitting.

        Returns:
            Pipeline: This object.
        """
        for pipe in self.pipes:
            pipe.fit(data)
            data = pipe.transform(data, is_inference=False)
        return self

    def transform(self, data: DataSet, is_inference: bool = False) -> DataSet:
        """Transforms the given data using this Pipeline.

        This Pipeline should be fitted first using fit().

        Args:
            data (DataSet): The DataSet to use in transforming.
            is_inference (Optional[bool]): Flag indicating if we're in inference
                mode for this transformation. We'll skip this step if
                skip_on_inference was set to True.

        Returns:
            DataSet: The Transformed Data.
        """
        for pipe in self.pipes:
            data = pipe.transform(data, is_inference=is_inference)
        return data

    def inverse_transform(self, data: DataSet, is_inference: bool = False) -> DataSet:
        """Inverse transforms the DataSet using this Pipeline.

        Note that this is automatically done in reverse: the inverse steps
        will be applied from back to front.

        Args:
            data (DataSet): The DataSet to use in inverse transforming.
            is_inference (Optional[bool]): Flag indicating if we're in inference
                mode for this transformation. We'll skip this step if
                skip_on_inference was set to True.

        Returns:
            DataSet: The inverse transformed DataSet
        """
        for pipe in reversed(self.pipes):
            data = pipe.inverse_transform(data, is_inference=is_inference)
        return data

    def reinitialise(self, args: Dict[str, Any]) -> None:
        """Re-initialises this Pipeline's Pipes using the given dictionary.

        Args:
            args (Dict[str, Any]): Dictionary containing new arguments.
                The keys of the `args` dict should be of the form `<name>__<arg_name>`:
                - name: the name of the Pipe.
                - arg_name: the argument name to use.
                The values will be used as argument parameters.
        """
        for pipe in self.pipes:
            if isinstance(pipe, Pipeline):
                pipe.reinitialise(args)
            else:
                pipe_args = get_args_for_prefix(f"{pipe.name}__", args)
                if len(pipe_args) > 0:
                    pipe.reinitialise(pipe_args)

    def copy(self, args: Optional[Dict[str, Any]] = None) -> "Pipeline":
        """Create a copy of this Pipeline using the given dictionary.

        Args:
            args (Dict[str, Any]): Dictionary containing new arguments.
                The keys of the `args` dict should be of the form `<name>__<arg_name>`:
                - name: the name of the Pipe.
                - arg_name: the argument name to use.
                The values will be used as argument parameters.
        """
        result = Pipeline([])
        for pipe in self.pipes:
            if isinstance(pipe, Pipeline):
                extra_pipeline = pipe.copy(args)
                result += extra_pipeline
            else:
                if args is None:
                    pipe_args = {}
                else:
                    pipe_args = get_args_for_prefix(f"{pipe.name}__", args)
                new_pipe = pipe.copy(pipe_args)
                result += new_pipe
        return result

    def __iter__(self) -> "Pipeline":
        """Prepares this object for iteration using next().

        Returns:
            Pipeline: this object.
        """
        self.__iter_n = 0
        return self

    def __next__(self) -> Union["Pipeline", Pipe]:
        """Gets the next item in this Pipeline.

        Raises:
            StopIteration: If this object has already been iterated.

        Returns:
            Union[Pipeline, Pipe]: The next Pipeline/Pipe.
        """
        if self.__iter_n >= len(self.pipes):
            raise StopIteration
        else:
            pos = self.__iter_n
            self.__iter_n += 1
            return self[pos]

    def __getitem__(self, pos: Union[int, str]) -> Union["Pipeline", Pipe]:
        """Gets the Pipe/Pipeline at the given position.

        Raises:
            KeyError: if `pos` does not exist in this object.
                int: `pos` is out of bounds.
                str: the given name does not match any Pipe in this object.
                    Any Pipelines in this object are also searched. This makes it
                    important to use different names for Pipes!

        Returns:
            Union[Pipeline, Pipe]: The Pipeline/Pipe matching the given position.
        """
        value = self._get(pos)
        if value is None:
            raise KeyError(f"{pos} is not found in this pipeline")
        return value

    def _get(self, pos: Union[int, str]) -> Union["Pipeline", Pipe, None]:
        if isinstance(pos, int):
            return self.pipes[pos]
        for pipe in self.pipes:
            if isinstance(pipe, Pipeline):
                result = pipe._get(pos)
                if result is not None:
                    return result
            elif pipe.name == pos:
                return pipe
        return None

    def __add__(self, obj: Union["Pipeline", Pipe]) -> "Pipeline":
        """Append a Pipe or Pipeline object and return a new Pipeline.

        Args:
            obj (Union[Pipeline, Pipe]): The object to be appended. Can be a Union[Pipeline, Pipe].

        Raises:
            TypeError: If `obj` is not a Pipe or Pipeline.

        Returns:
            Pipeline: A Pipeline with the two sets of steps combined. The steps from this
                object will always be used first.
        """
        if isinstance(obj, Pipeline):
            return Pipeline(self.pipes + obj.pipes)
        elif isinstance(obj, Pipe):
            return Pipeline(self.pipes + [obj])
        else:
            raise TypeError("Pipeline can only use + with a Pipe or Pipeline")

    def insert(self, pos: int, pipe: Pipe) -> None:
        """Insert a pipe in the given integer position.

        We check if the pipe name doesn't exist yet.

        Args:
            pos (int): The position to put the new pipe
            pipe (Pipe): The pipe to insert
        """
        self._assert_all_names_different(set(pipe.name))
        self.pipes.insert(pos, pipe)

    def __len__(self) -> int:
        """Return the length of this Pipeline.

        If it contains any sub-pipelines, those will be counted in full as well
        (and thus not as a single step, unless they only contain 1 step).

        Returns:
            int: The lenght of this Pipeline.
        """
        result = 0
        for p in self.pipes:
            if isinstance(p, Pipeline):
                result += len(p)
            else:
                result += 1
        return result

    def __str__(self, indents: int = 0) -> str:
        """Create string representation of this Pipeline.

        Args:
            indents (int, optional): The number of preceding tabs. Defaults to 0.

        Returns:
            str: A string representation of this Pipeline.
        """
        tabs = "\t" * indents
        output = [tabs + "Pipeline:"]
        indents += 1
        for pipe in self.pipes:
            if isinstance(pipe, Pipe):
                output.append(tabs + str(pipe))
            else:
                output.append(pipe.__str__(indents))
        return "\n".join(output)
