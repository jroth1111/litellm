import importlib.util
import pytest


_REQUIRED_MODULES = ("apscheduler", "fastapi", "orjson", "yaml", "backoff")
_missing = [name for name in _REQUIRED_MODULES if importlib.util.find_spec(name) is None]

if _missing:
    pytest.skip(
        f"Missing proxy dependencies: {', '.join(_missing)}",
        allow_module_level=True,
    )
