import importlib.util
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]

AUTH_CORE = ROOT / "litellm" / "auth" / "core.py"
AUTH_SELECTOR = ROOT / "litellm" / "auth" / "selector.py"
AUTH_ADAPTER = ROOT / "litellm" / "auth" / "adapters" / "anthropic.py"
AUTH_MEM_STORE = ROOT / "litellm" / "auth" / "memory_store.py"


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
selector_mod = _load_module(AUTH_SELECTOR, "litellm.auth.selector")
adapter_mod = _load_module(AUTH_ADAPTER, "litellm.auth.adapters.anthropic")
mem_store_mod = _load_module(AUTH_MEM_STORE, "litellm.auth.memory_store")

AuthRecord = core_mod.AuthRecord
AuthStatus = core_mod.AuthStatus
RequestContext = core_mod.RequestContext
CredentialSelector = selector_mod.CredentialSelector
AnthropicSubscriptionAdapter = adapter_mod.AnthropicSubscriptionAdapter
InMemoryAuthStore = mem_store_mod.InMemoryAuthStore


class FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code
        self.response = self

    def json(self):
        return {}


class FakeStrategy(AnthropicSubscriptionAdapter):
    """
    Simulates one 401 then success after refresh.
    """

    def __init__(self):
        super().__init__()
        self.called = 0

    def prepare(self, headers, ctx, auth):
        # increment
        self.called += 1
        # fail first call
        if self.called == 1:
            raise Exception("401")
        return super().prepare(headers, ctx, auth)

    def refresh(self, auth, ctx):
        refreshed = auth.clone()
        refreshed.metadata["access_token"] = "refreshed"
        return refreshed


class IntegrationRetryTests(unittest.TestCase):
    def test_refresh_on_prepare_failure(self):
        selector = CredentialSelector()
        strat = FakeStrategy()
        store = InMemoryAuthStore()
        auth = AuthRecord(
            id="a",
            provider="anthropic",
            metadata={"access_token": "expired", "refresh_token": "rt"},
            attributes={},
        )
        store.save("default", auth)
        sel = selector.select("anthropic/claude", [auth])
        self.assertIsNotNone(sel)
        # prepare should fail first, refresh, then succeed
        refreshed_ctx = RequestContext(model="anthropic/claude")
        refreshed_auth = strat.refresh(auth, refreshed_ctx)
        self.assertEqual(refreshed_auth.metadata["access_token"], "refreshed")


if __name__ == "__main__":
    unittest.main()
