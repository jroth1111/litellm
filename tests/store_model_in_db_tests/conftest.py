import importlib.util
import pytest


missing_deps = [
    dep
    for dep in ("assemblyai", "multipart", "fastapi", "apscheduler")
    if importlib.util.find_spec(dep) is None
]
if missing_deps:
    pytest.skip(
        f"store_model_in_db tests require dependencies: {', '.join(missing_deps)}",
        allow_module_level=True,
    )
