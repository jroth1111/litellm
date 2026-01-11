import importlib.util
import pytest


if importlib.util.find_spec("google") is None or importlib.util.find_spec("google.genai") is None:
    pytest.skip(
        "unified google tests require google genai dependencies",
        allow_module_level=True,
    )
