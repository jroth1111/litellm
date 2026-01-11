import os
import pytest


if os.getenv("LITELLM_RUN_OPENAI_ENDPOINTS_TESTS", "").lower() not in ("1", "true", "yes"):
    pytest.skip(
        "openai endpoints tests disabled; set LITELLM_RUN_OPENAI_ENDPOINTS_TESTS=1 to run",
        allow_module_level=True,
    )
