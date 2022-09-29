from pathlib import Path

from fastapi import FastAPI

from pype.base.deploy import Inferencer, PypeApp


def load_model() -> Inferencer:
    """Wrapper function to load an Inferencer from a fixed location in a package.

    Returns:
        Inferencer: The Inferencer contained in this package.
    """
    folder = Path(__file__).parent / "outputs"
    print(f"Loading inferencer from {str(folder)}")
    return Inferencer.from_folder(folder)


def load_app() -> FastAPI:
    """Wrapper function to create a FastAPI server from a fixed location in this package.

    Based on PypeApp.

    If you create a python file F where this App is loaded:
    ```
    from <model name> import load_app
    app = load_app()
    ```

    You can host this model using a command like:
    uvicorn <python package path to file F>:app

    Returns:
        FastAPI: The FastAPI/PypeApp server used to host the model.
    """
    folder = Path(__file__).parent / "outputs"
    return PypeApp("model", folder=folder).create_app()
