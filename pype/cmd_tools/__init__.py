"""Only intended for command line tools."""
from argparse import ArgumentParser

from .dev_tools import run_dev_install


def main() -> None:
    """Main console entrypoint for pype.

    Raises:
        ValueError: If the subcommand is not allowed. Current values are:
            - dev-install
    """
    allowed_values = ["dev-install"]

    parser = ArgumentParser()
    parser.add_argument("command", type=str, help=f"The command to run. Allowed values: {allowed_values}")

    parsed, _ = parser.parse_known_args()
    command = parsed.command

    if command not in allowed_values:
        raise ValueError(f"`{command}` is not a valid pype command")

    if command == "dev-install":
        run_dev_install(parser)
