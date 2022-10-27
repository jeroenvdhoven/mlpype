"""Only intended for command line tools for development."""
from argparse import ArgumentParser

from .dev_tools import run_dev_build, run_dev_install
from .helpers import split_cmd_line


def main() -> None:
    """Main console entrypoint for pype.

    Raises:
        ValueError: If the subcommand is not allowed. Current values are:
            - dev install
            - dev build
    """
    allowed_values = {
        "dev": {
            "install": run_dev_install,
            "build": run_dev_build,
        }
    }

    parser = ArgumentParser()
    split_cmd_line(parser, allowed_values, level=0)  # type: ignore
