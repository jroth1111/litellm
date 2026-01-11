import importlib.util
import os
import pytest


missing_deps = [
    dep for dep in ("fastapi", "apscheduler") if importlib.util.find_spec(dep) is None
]
if missing_deps or os.getenv("LITELLM_RUN_PROXY_SECURITY_TESTS", "").lower() not in (
    "1",
    "true",
    "yes",
):
    pytest.skip(
        "proxy security tests disabled or missing proxy dependencies",
        allow_module_level=True,
    )
