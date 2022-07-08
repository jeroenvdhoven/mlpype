import inspect
import typing
import warnings
from argparse import ArgumentParser
from typing import Callable, Iterable, Type

from pype.base.pipeline.pipe import Pipe
from pype.base.pipeline.pipeline import Pipeline


def add_args_to_parser_for_pipeline(parser: ArgumentParser, pipeline: Pipeline) -> None:
    """Adds arguments for a full pipeline re-initialisation to the given ArgumentParser.

    This will add arguments for every Pipe in the pipeline.

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
    excluded_superclasses: list[Type],
    excluded_args: list[str] | None = None,
) -> None:
    """Adds arguments for the constructor of a class to the given parser.

    We try to dive deeper in case of variable kw-args, but in general we recommend not
    using these.

    Args:
        parser (ArgumentParser): The ArgumentParser to add arguments to.
        class_ (Type): The class who's constructor's arguments we want to add.
        prefix (str): A prefix that should be set on each argument name before adding it to
            the ArgumentParser.
        excluded_superclasses (list[Type]): Superclasses of the class_ that should be ignored in case
            there are kw-args style arguments to the constructor.
        excluded_args (list[str] | None, optional): argument names to never include. By default 'self' and 'cls' are ignored.
    """
    init_func = class_.__init__
    add_args_to_parser_for_function(parser, init_func, prefix, excluded_args)

    signature = inspect.signature(init_func)
    for _, param in signature.parameters.items():
        # we only need to check for superclass arguments if there are var-args.
        # we ignore positional var-args.
        if param.kind == param.VAR_KEYWORD:
            for superclass in class_.__bases__:
                if superclass not in excluded_superclasses:
                    add_args_to_parser_for_class(parser, superclass, prefix, excluded_superclasses, excluded_args)


def add_args_to_parser_for_function(
    parser: ArgumentParser, function: Callable, prefix: str, excluded: list[str] | None = None
) -> None:
    """Add arguments for a given function to the given parser.

    Args:
        parser (ArgumentParser): The ArgumentParser to add arguments to.
        function (Callable): The function who's arguments we want to add.
        prefix (str): A prefix that should be set on each argument name before adding it to
            the ArgumentParser.
        excluded (list[str] | None, optional): argument names to never include. By default 'self' and 'cls' are ignored.
    """
    args = inspect.signature(function)
    for name, parameter in args.parameters.items():
        class_ = parameter.annotation
        required = parameter.default == inspect._empty
        add_argument(parser, name, prefix, class_, required, excluded)


def add_argument(
    parser: ArgumentParser, name: str, prefix: str, class_: type, is_required: bool, excluded: list[str] | None = None
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
        excluded (list[str] | None, optional): argument names to never include. By default 'self' and 'cls' are ignored.
    """
    if excluded is None:
        excluded = []
    excluded = excluded + ["self", "cls"]

    arg_name = f"--{prefix}__{name}"
    if name in excluded:
        return
    if class_ in [str, float, int, bool]:
        parser.add_argument(arg_name, type=class_, required=is_required)
    elif typing.get_origin(class_) in [list, tuple, Iterable]:
        subtype = typing.get_args(class_)[0]
        parser.add_argument(arg_name, type=subtype, nargs="+", required=is_required)
    else:
        warnings.warn(f"Currently the class {str(class_)} is not supported for automatic command line importing.")
