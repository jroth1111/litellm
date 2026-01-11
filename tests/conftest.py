import importlib.util
import inspect

import pytest


PYTEST_ASYNCIO_AVAILABLE = importlib.util.find_spec("pytest_asyncio") is not None


def pytest_collection_modifyitems(config, items):
    if PYTEST_ASYNCIO_AVAILABLE:
        return

    skip_async = pytest.mark.skip(reason="pytest-asyncio not installed")
    for item in items:
        if item.get_closest_marker("asyncio") is not None:
            item.add_marker(skip_async)
            continue
        func = getattr(item, "obj", None)
        if func is not None and inspect.iscoroutinefunction(func):
            item.add_marker(skip_async)
