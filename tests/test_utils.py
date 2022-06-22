from contextlib import contextmanager


@contextmanager
def pytest_assert(error, message: str):
    try:
        yield
        raise ValueError("No error was raised!")
    except error as e:
        assert e.args[0] == message
