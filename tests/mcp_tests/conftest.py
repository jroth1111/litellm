import importlib.util
import pytest


spec = importlib.util.find_spec("mcp")
if spec is None:
    pytest.skip("mcp tests require the mcp package", allow_module_level=True)
else:
    import mcp

    required_attrs = [
        "ClientSession",
        "ReadResourceResult",
        "Resource",
        "StdioServerParameters",
    ]
    missing_attrs = [name for name in required_attrs if not hasattr(mcp, name)]
    if missing_attrs:
        pytest.skip(
            f"mcp tests require newer mcp package with {', '.join(missing_attrs)}",
            allow_module_level=True,
        )
