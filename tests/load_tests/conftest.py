import os
import pytest


if os.getenv("LITELLM_RUN_LOAD_TESTS", "").lower() not in ("1", "true", "yes"):
    pytest.skip(
        "load tests disabled; set LITELLM_RUN_LOAD_TESTS=1 to run",
        allow_module_level=True,
    )
