from pathlib import Path

from fastapi.applications import FastAPI

from pype.fastapi.deploy import PypeApp


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
    folder = Path(__file__).parent.parent / "outputs"
    return PypeApp("model", folder=folder).create_app()
