import importlib.util
import pytest


missing_deps = [
    dep
    for dep in (
        "anthropic",
        "assemblyai",
        "langchain_mcp_adapters",
        "vertexai",
    )
    if importlib.util.find_spec(dep) is None
]
if missing_deps:
    pytest.skip(
        f"pass-through tests require optional dependencies: {', '.join(missing_deps)}",
        allow_module_level=True,
    )
