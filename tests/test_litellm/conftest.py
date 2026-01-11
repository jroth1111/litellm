# conftest.py

import importlib
import importlib.util
import os
import sys

import pytest

missing_deps = [
    dep
    for dep in ("respx", "opentelemetry", "polars", "openapi_core", "botocore")
    if importlib.util.find_spec(dep) is None
]
spec = importlib.util.find_spec("mcp")
missing_mcp = spec is None
missing_mcp_attrs = False
if spec is not None:
    import mcp

    required_attrs = ["ClientSession", "ReadResourceResult", "Resource"]
    missing_mcp_attrs = any(not hasattr(mcp, name) for name in required_attrs)

SKIP_TEST_LITELLM = bool(missing_deps or missing_mcp or missing_mcp_attrs)

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio

import litellm


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()




@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    curr_dir = os.getcwd()  # Get the current working directory
    sys.path.insert(
        0, os.path.abspath("../..")
    )  # Adds the project directory to the system path

    import litellm
    from litellm import Router

    importlib.reload(litellm)

    try:
        if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
            import litellm.proxy.proxy_server

            importlib.reload(litellm.proxy.proxy_server)
    except Exception as e:
        print(f"Error reloading litellm.proxy.proxy_server: {e}")

    litellm.in_memory_llm_clients_cache.flush_cache()

    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    print(litellm)
    # from litellm import Router, completion, aembedding, acompletion, embedding
    yield

    # Teardown code (executes after the yield point)
    loop.close()  # Close the loop created earlier
    asyncio.set_event_loop(None)  # Remove the reference to the loop


def pytest_collection_modifyitems(config, items):
    # Separate tests in 'test_amazing_proxy_custom_logger.py' and other tests
    custom_logger_tests = [
        item for item in items if "custom_logger" in item.parent.name
    ]
    other_tests = [item for item in items if "custom_logger" not in item.parent.name]

    # Sort tests based on their names
    custom_logger_tests.sort(key=lambda x: x.name)
    other_tests.sort(key=lambda x: x.name)

    # Reorder the items list
    items[:] = custom_logger_tests + other_tests


def pytest_ignore_collect(path, config):
    if SKIP_TEST_LITELLM:
        return True
    return False
