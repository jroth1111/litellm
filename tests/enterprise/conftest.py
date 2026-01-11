import importlib.util
import pytest


missing_deps = [
    dep
    for dep in ("fastapi", "apscheduler", "prometheus_client")
    if importlib.util.find_spec(dep) is None
]
if missing_deps:
    pytest.skip(
        f"enterprise tests require optional dependencies: {', '.join(missing_deps)}",
        allow_module_level=True,
    )
