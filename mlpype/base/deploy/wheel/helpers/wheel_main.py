from pathlib import Path

from mlpype.base.deploy import Inferencer


def load_model() -> Inferencer:
    """Wrapper function to load an Inferencer from a fixed location in a package.

    Returns:
        Inferencer: The Inferencer contained in this package.
    """
    folder = Path(__file__).parent.parent / "outputs"
    print(f"Loading inferencer from {str(folder)}")
    return Inferencer.from_folder(folder)
