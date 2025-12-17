import importlib.util
import pathlib
import sys
import types
import unittest

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUTH_CORE = ROOT / "litellm" / "auth" / "core.py"
AUTH_ADAPTER = ROOT / "litellm" / "auth" / "adapters" / "anthropic.py"


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# scaffold package paths
if "litellm" not in sys.modules:
    pkg = types.ModuleType("litellm")
    pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
    sys.modules["litellm"] = pkg
if "litellm.auth" not in sys.modules:
    pkg = types.ModuleType("litellm.auth")
    pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
    sys.modules["litellm.auth"] = pkg
if "litellm.auth.adapters" not in sys.modules:
    pkg = types.ModuleType("litellm.auth.adapters")
    pkg.__path__ = [str(ROOT / "litellm" / "auth" / "adapters")]  # type: ignore[attr-defined]
    sys.modules["litellm.auth.adapters"] = pkg
if "litellm.llms" not in sys.modules:
    pkg = types.ModuleType("litellm.llms")
    pkg.__path__ = [str(ROOT / "litellm" / "llms")]  # type: ignore[attr-defined]
    sys.modules["litellm.llms"] = pkg
if "litellm.llms.anthropic" not in sys.modules:
    pkg = types.ModuleType("litellm.llms.anthropic")
    pkg.__path__ = [str(ROOT / "litellm" / "llms" / "anthropic")]  # type: ignore[attr-defined]
    sys.modules["litellm.llms.anthropic"] = pkg

core_mod = _load_module(AUTH_CORE, "litellm.auth.core")
adapter_mod = _load_module(AUTH_ADAPTER, "litellm.auth.adapters.anthropic")

AuthRecord = core_mod.AuthRecord
RequestContext = core_mod.RequestContext
AnthropicSubscriptionAdapter = adapter_mod.AnthropicSubscriptionAdapter


class MockTransport(httpx.MockTransport):
    def __init__(self, payload: dict):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        super().__init__(handler)


class MockedOAuthFlowTests(unittest.TestCase):
    def test_refresh_with_mocked_oauth_endpoint(self):
        strat = AnthropicSubscriptionAdapter()
        auth = AuthRecord(
            id="a",
            provider="anthropic",
            metadata={"refresh_token": "rt-old"},
            attributes={"token_url": "https://mock/oauth/token"},
        )
        payload = {
            "access_token": "new-token",
            "refresh_token": "new-rt",
            "expires_in": 120,
        }
        transport = MockTransport(payload)

        orig_client = httpx.Client

        def fake_client(*args, **kwargs):
            # bypass recursive patch
            return orig_client(transport=transport, timeout=5.0)

        httpx.Client = fake_client  # type: ignore
        try:
            refreshed = strat.refresh(auth, RequestContext(model="anthropic/claude"))
        finally:
            httpx.Client = orig_client  # type: ignore

        self.assertEqual(refreshed.metadata["access_token"], "new-token")
        self.assertEqual(refreshed.metadata["refresh_token"], "new-rt")
        self.assertIn("expires_at", refreshed.metadata)


if __name__ == "__main__":
    unittest.main()
