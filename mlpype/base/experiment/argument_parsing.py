import inspect
import re
import typing
import warnings
from argparse import ArgumentParser
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

from docstring_parser import parse

from mlpype.base.pipeline.pipe import Pipe
from mlpype.base.pipeline.pipeline import Pipeline


def add_args_to_parser_for_pipeline(parser: ArgumentParser, pipeline: Pipeline) -> None:
    """Adds arguments for a full pipeline re-initialisation to the given ArgumentParser.

    This will add arguments for every Pipe in the pipeline.

    Arguments will be drawn in order from:
        - type hints
        - documentation.

    Documentation hints are not guaranteed to work. For the most reliable results, use type hints.

    Args:
        parser (ArgumentParser): The ArgumentParser to add arguments to.
        pipeline (Pipeline): The pipeline for which re-initialisation arguments should be added.
    """
    for pipe in pipeline:
        if isinstance(pipe, Pipe):
            add_args_to_parser_for_class(parser, pipe.operator_class, f"pipeline__{pipe.name}", [])
        else:
            add_args_to_parser_for_pipeline(parser, pipe)


def add_args_to_parser_for_class(
    parser: ArgumentParser,
    class_: Type,
    prefix: str,
    excluded_superclasses: List[Type],
    excluded_args: Optional[List[str]] = None,
) -> None:
    """Adds arguments for the constructor of a class to the given parser.

    We try to dive deeper in case of variable kw-args, but in general we recommend not
    using these.

    Arguments will be drawn in order from:
        - type hints
        - documentation.

    Documentation hints are not guaranteed to work. For the most reliable results, use type hints.

    Args:
        parser (ArgumentParser): The ArgumentParser to add arguments to.
        class_ (Type): The class who's constructor's arguments we want to add.
        prefix (str): A prefix that should be set on each argument name before adding it to
            the ArgumentParser.
        excluded_superclasses (List[Type]): Superclasses of the class_ that should be ignored in case
            there are kw-args style arguments to the constructor.
        excluded_args (Optional[List[str]], optional): argument names to never include.
            By default 'self' and 'cls' are ignored.
    """
    init_func = class_.__init__
    class_docstring_args = _parse_docs_to_type_args(class_)
    if len(class_docstring_args) == 0:
        class_docstring_args = _parse_docs_to_type_args(init_func)

    add_args_to_parser_for_function(parser, init_func, prefix, excluded_args, class_docstring_args=class_docstring_args)

    signature = inspect.signature(init_func)
    for _, param in signature.parameters.items():
        # we only need to check for superclass arguments if there are var-args.
        # we ignore positional var-args.
        if param.kind == param.VAR_KEYWORD:
            for superclass in class_.__bases__:
                if superclass not in excluded_superclasses:
                    add_args_to_parser_for_class(parser, superclass, prefix, excluded_superclasses, excluded_args)


def add_args_to_parser_for_function(
    parser: ArgumentParser,
    function: Callable,
    prefix: str,
    excluded: Optional[List[str]] = None,
    class_docstring_args: Optional[Dict[str, Union[type, None]]] = None,
) -> None:
    """Add arguments for a given function to the given parser.

    Args:
        parser (ArgumentParser): The ArgumentParser to add arguments to.
        function (Callable): The function who's arguments we want to add.
        prefix (str): A prefix that should be set on each argument name before adding it to
            the ArgumentParser.
        excluded (Optional[List[str]]): argument names to never include. By default 'self' and 'cls' are ignored.
        class_docstring_args: (Optional[Dict[str, Union[type, None]]): docstring arguments obtained from the class.
            Useful for init functions, since those can be documented in multiple places.
    """
    args = inspect.signature(function)
    docstring_args = _parse_docs_to_type_args(function)

    if class_docstring_args is not None:
        docstring_args.update(**class_docstring_args)

    for name, parameter in args.parameters.items():
        class_ = parameter.annotation
        if class_ == inspect._empty and name in docstring_args:
            class_ = docstring_args[name]
        required = parameter.default == inspect._empty
        add_argument(parser, name, prefix, class_, required, parameter.default, excluded)


def _parse_docs_to_type_args(
    func: Callable, extra_mappings: Optional[Dict[str, type]] = None, include_none_args: bool = False
) -> Dict[str, Union[type, None]]:
    """Parses the docstring of a function into an arg-type dictionary.

    We only parse the following types, whose first letter can be capitalized.
        - bool
        - str
        - int
        - float

    This is all done on a best-effort basis. We do not guarantee this will work for every
    function in every library. For the best results, make sure you use proper type hinting
    instead.

    Args:
        func (Callable): The function whose docstring needs to be read.
        extra_mappings (Dict[str, type], optional): Optional extra regex -> type mappings.
            Defaults to no extra mappings.
        include_none_args (bool, optional): Whether to include arguments in the output
            with no good type mappings. Defaults to False.

    Returns:
        Dict[str, Union[type, None]]: A dictionary mapping argument names to types.
    """
    signature = inspect.signature(func)
    if extra_mappings is None:
        extra_mappings = {}

    result: Dict[str, Union[type, None]] = {}
    doc = inspect.getdoc(func)
    if doc is None:
        return result

    parsed_doc = parse(doc)
    for param in parsed_doc.params:
        if param.arg_name in signature.parameters:
            class_ = _parse_type_name(param.type_name, extra_mappings)
            if include_none_args or class_ is not None:
                result[param.arg_name] = class_

    return result


def _parse_type_name(s: Optional[str], extra_mappings: Optional[Dict[str, type]] = None) -> Union[type, None]:
    if extra_mappings is None:
        extra_mappings = {}
    if s is None:
        return None

    regexes = {
        r"(^|\W)[fF]loat($|\W)": float,
        r"(^|\W)[sS]tr($|\W)": str,
        r"(^|\W)[bB]ool($|\W)": bool,
        r"(^|\W)[iI]nt($|\W)": int,
    }
    regexes.update(**extra_mappings)

    for regex, class_ in regexes.items():
        if re.search(regex, s) is not None:
            return class_
    return None


def add_argument(
    parser: ArgumentParser,
    name: str,
    prefix: str,
    class_: Union[type, None],
    is_required: bool,
    default: Any,
    excluded: Optional[List[str]] = None,
) -> None:
    """Add an argument to the given parser.

    Args:
        parser (ArgumentParser): The ArgumentParser to add arguments to.
        name (str): The name of the argument
        prefix (str): A prefix that should be set on each argument name before adding it to
            the ArgumentParser.
        class_ (type): The class of the argument. Currently we support:
            - str, float, int, bool
            - lists / tuples / iterables of the above 4 arguments.
        is_required (bool): Whether the argument is required or optional.
        default (Any): the default value if is_required is False.
        excluded (Optional[List[str]]): argument names to never include. By default 'self' and 'cls' are ignored.
    """
    if excluded is None:
        excluded = []
    excluded = excluded + ["self", "cls"]

    if not is_required:
        assert default != inspect._empty, "A default value must be provided if the argument is not required"
    else:
        default = None

    arg_name = f"--{prefix}__{name}"
    if name in excluded or class_ is None:
        return
    if class_ in [str, float, int, bool]:
        parser.add_argument(arg_name, type=_get_conversion_function(class_), required=is_required, default=default)
    elif typing.get_origin(class_) in [
        list,
        tuple,
        typing.get_origin(Iterable),
        typing.get_origin(List),
        typing.get_origin(Tuple),
    ]:
        subtype = typing.get_args(class_)[0]
        parser.add_argument(
            arg_name, type=_get_conversion_function(subtype), nargs="+", required=is_required, default=default
        )
    else:
        warnings.warn(f"Currently the class {str(class_)} is not supported for automatic command line importing.")


def _get_conversion_function(class_: type) -> Callable:
    if class_ == bool:
        return _convert_bool
    return class_


def _convert_bool(s: str) -> bool:
    return s.lower() in ["true", "1"]
