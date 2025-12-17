import importlib.util
import pathlib
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUTH_CORE = ROOT / "litellm" / "auth" / "core.py"
AUTH_ADAPTER = ROOT / "litellm" / "auth" / "adapters" / "anthropic.py"
AUTH_MEM_STORE = ROOT / "litellm" / "auth" / "memory_store.py"


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# Scaffold minimal package path to avoid importing litellm/__init__.py
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
mem_store_mod = _load_module(AUTH_MEM_STORE, "litellm.auth.memory_store")

AuthRecord = core_mod.AuthRecord
AuthStatus = core_mod.AuthStatus
RequestContext = core_mod.RequestContext
InMemoryAuthStore = mem_store_mod.InMemoryAuthStore
AnthropicSubscriptionAdapter = adapter_mod.AnthropicSubscriptionAdapter


class FakeHttpError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class MiniRouter:
    """
    Lightweight shim of the router refresh path: prepare -> call -> on 429 refresh -> retry once.
    """

    def __init__(self, store: InMemoryAuthStore):
        self.store = store

    def _maybe_refresh_auth(self, auth: AuthRecord, strategy, ctx: RequestContext) -> AuthRecord:
        refreshed = strategy.refresh(auth, ctx)
        self.store.save("default", refreshed)
        return refreshed

    def _prepare(self, strategy, auth: AuthRecord, model: str) -> AuthRecord:
        ctx = RequestContext(model=model)
        exp = strategy.expiration(auth) or auth.expiration_time()
        lead = strategy.refresh_lead(auth)
        now = datetime.now(timezone.utc)
        if exp and lead and exp - now <= lead:
            auth = self._maybe_refresh_auth(auth, strategy, ctx)
        strategy.prepare({}, ctx, auth)
        return auth

    def call(self, strategy, auth: AuthRecord, model: str = "anthropic/claude"):
        ctx = RequestContext(model=model)
        auth = self._prepare(strategy, auth, model)
        try:
            self._do_call(auth)
        except FakeHttpError as exc:
            if exc.status_code in (401, 403, 429):
                auth = self._maybe_refresh_auth(auth, strategy, ctx)
                self._do_call(auth)
            else:
                raise
        return auth

    def _do_call(self, auth: AuthRecord):
        if auth.metadata.get("access_token") == "expired":
            raise FakeHttpError(429)
        return "ok"


class RouterRefreshOn429Tests(unittest.TestCase):
    def test_refresh_on_429_persists_and_retries(self):
        strat = AnthropicSubscriptionAdapter()
        store = InMemoryAuthStore()
        auth = AuthRecord(
            id="a",
            provider="anthropic",
            metadata={"access_token": "expired", "refresh_token": "rt"},
            attributes={},
            status=AuthStatus.ACTIVE,
        )
        store.save("default", auth)

        class RefreshOnceStrategy(AnthropicSubscriptionAdapter):
            def refresh(self, auth, ctx):  # type: ignore[override]
                updated = auth.clone()
                updated.metadata["access_token"] = "fresh"
                updated.last_refreshed_at = datetime.now(timezone.utc)
                return updated

        strat = RefreshOnceStrategy()
        router = MiniRouter(store)

        refreshed = router.call(strat, auth)
        persisted = store.get("default", "a")

        self.assertEqual(refreshed.metadata["access_token"], "fresh")
        self.assertIsNotNone(refreshed.last_refreshed_at)
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.metadata["access_token"], "fresh")


if __name__ == "__main__":
    unittest.main()
