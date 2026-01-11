import importlib.util
import pytest


missing_deps = [
    dep
    for dep in ("boto3", "respx", "prisma", "opentelemetry")
    if importlib.util.find_spec(dep) is None
]
if missing_deps:
    pytest.skip(
        f"logging callback tests require optional dependencies: {', '.join(missing_deps)}",
        allow_module_level=True,
    )
