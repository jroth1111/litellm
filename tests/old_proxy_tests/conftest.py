import os
import pytest


if os.getenv("LITELLM_RUN_OLD_PROXY_TESTS", "").lower() not in ("1", "true", "yes"):
    pytest.skip(
        "old proxy tests disabled; set LITELLM_RUN_OLD_PROXY_TESTS=1 to run",
        allow_module_level=True,
    )
