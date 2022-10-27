"""Only intended for command line tools."""
from argparse import ArgumentParser
from types import FunctionType
from typing import Callable

NestedCommands = dict[str, "NestedCommands" | Callable[[ArgumentParser], None]]  # type: ignore


def split_cmd_line(
    parser: ArgumentParser,
    allowed_values: NestedCommands,
    level: int = 0,
) -> None:
    """Split the command line flow along fixed commands.

    Raises:
        ValueError: If the subcommand is not allowed. Current values are:
            - dev-install
    """
    arg_name = "_".join(["sub"] * level + ["command"])

    allowed_commands = _get_allowed_commands(allowed_values)
    parser.add_argument(
        "command", type=str, help=f"The {arg_name.replace('_', '-')} to run. Allowed values:\n{allowed_commands}"
    )

    parsed, _ = parser.parse_known_args()
    command = parsed.command

    if command not in allowed_values:
        raise ValueError(f"`{command}` is not a valid pype command")

    sub_call = allowed_values[command]
    if isinstance(sub_call, dict):
        split_cmd_line(parser, sub_call, level + 1)
    elif isinstance(sub_call, FunctionType):
        sub_call(parser)
    else:
        raise ValueError(f"Type `{type(sub_call)}` is not allowed: only callables and dicts are allowed.")


def _get_allowed_commands(allowed_values: NestedCommands) -> str:
    return ", ".join(__get_allowed_commands(allowed_values))


def __get_allowed_commands(allowed_values: NestedCommands) -> list[str]:
    res = []
    for name, values in allowed_values.items():
        if isinstance(values, dict):
            commands = [f"{name} {c}" for c in __get_allowed_commands(values)]
            res.extend(commands)
        else:
            res.append(name)
    return res
