import datetime
import importlib.util
import pathlib
import sys
import types
import unittest
from typing import Any

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUTH_CORE = ROOT / "litellm" / "auth" / "core.py"
ADAPTER = ROOT / "litellm" / "auth" / "adapters" / "anthropic.py"


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# Minimal package scaffolding to satisfy relative imports
if "litellm" not in sys.modules:
    litellm_pkg = types.ModuleType("litellm")
    litellm_pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
    sys.modules["litellm"] = litellm_pkg
if "litellm.auth" not in sys.modules:
    auth_pkg = types.ModuleType("litellm.auth")
    auth_pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
    sys.modules["litellm.auth"] = auth_pkg
if "litellm.auth.adapters" not in sys.modules:
    adapters_pkg = types.ModuleType("litellm.auth.adapters")
    adapters_pkg.__path__ = [str(ROOT / "litellm" / "auth" / "adapters")]  # type: ignore[attr-defined]
    sys.modules["litellm.auth.adapters"] = adapters_pkg
if "litellm.llms" not in sys.modules:
    llms_pkg = types.ModuleType("litellm.llms")
    llms_pkg.__path__ = [str(ROOT / "litellm" / "llms")]  # type: ignore[attr-defined]
    sys.modules["litellm.llms"] = llms_pkg
if "litellm.llms.anthropic" not in sys.modules:
    anthropic_pkg = types.ModuleType("litellm.llms.anthropic")
    anthropic_pkg.__path__ = [str(ROOT / "litellm" / "llms" / "anthropic")]  # type: ignore[attr-defined]
    sys.modules["litellm.llms.anthropic"] = anthropic_pkg

core_mod = _load_module(AUTH_CORE, "litellm.auth.core")
sys.modules["litellm.auth.core"] = core_mod  # ensure relative import resolution
sys.modules.setdefault("litellm.auth", sys.modules.get("litellm.auth", types.ModuleType("litellm.auth")))  # type: ignore
sys.modules["litellm.auth"].__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
adapter_mod = _load_module(ADAPTER, "litellm.auth.adapters.anthropic")

AuthRecord = core_mod.AuthRecord
AuthStatus = core_mod.AuthStatus
RequestContext = core_mod.RequestContext
AnthropicSubscriptionAdapter = adapter_mod.AnthropicSubscriptionAdapter


class DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class DummyClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.called = False

    def __enter__(self) -> "DummyClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def post(self, *_args, **_kwargs) -> DummyResponse:
        self.called = True
        return DummyResponse(self.payload)


class StrategyRefreshTests(unittest.TestCase):
    def test_refresh_updates_access_token_and_expiry(self):
        strat = AnthropicSubscriptionAdapter()
        auth = AuthRecord(
            id="a",
            provider="anthropic",
            metadata={"refresh_token": "rt-old"},
            attributes={},
        )

        payload = {
            "access_token": "new-token",
            "refresh_token": "new-rt",
            "expires_in": 3600,
        }

        def fake_client(*_args, **_kwargs):
            return DummyClient(payload)

        # monkeypatch httpx.Client to avoid network
        orig_client = httpx.Client
        httpx.Client = fake_client  # type: ignore
        try:
            refreshed = strat.refresh(auth, RequestContext(model="anthropic/claude"))
        finally:
            httpx.Client = orig_client  # type: ignore

        self.assertEqual(refreshed.metadata["access_token"], "new-token")
        self.assertEqual(refreshed.metadata["refresh_token"], "new-rt")
        self.assertEqual(refreshed.status, AuthStatus.ACTIVE)
        self.assertFalse(refreshed.unavailable)
        self.assertEqual(refreshed.status_message, "")
        self.assertIn("expires_at", refreshed.metadata)
        exp = refreshed.metadata["expires_at"]
        self.assertIsInstance(exp, str)
        # parseable
        datetime.datetime.fromisoformat(exp)


if __name__ == "__main__":
    unittest.main()
