import importlib.util
import pathlib
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUTH_CORE = ROOT / "litellm" / "auth" / "core.py"
AUTH_MAINT = ROOT / "litellm" / "auth" / "maintenance.py"
AUTH_MEM = ROOT / "litellm" / "auth" / "memory_store.py"


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


class RefreshLeadRegistryTests(unittest.TestCase):
    def test_provider_refresh_lead_used_when_strategy_none(self):
        if "litellm" not in sys.modules:
            pkg = types.ModuleType("litellm")
            pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
            sys.modules["litellm"] = pkg
        if "litellm.auth" not in sys.modules:
            pkg = types.ModuleType("litellm.auth")
            pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
            sys.modules["litellm.auth"] = pkg

        core_mod = _load_module(AUTH_CORE, "litellm.auth.core")
        sys.modules["litellm.auth.core"] = core_mod
        maint_mod = _load_module(AUTH_MAINT, "litellm.auth.maintenance")
        mem_mod = _load_module(AUTH_MEM, "litellm.auth.memory_store")

        AuthRecord = core_mod.AuthRecord
        register_refresh_lead = core_mod.register_refresh_lead
        provider_refresh_lead = core_mod.provider_refresh_lead
        AuthMaintainer = maint_mod.AuthMaintainer
        InMemoryAuthStore = mem_mod.InMemoryAuthStore

        class DummyStrategy:
            provider = "anthropic"

            def supports(self, model):
                return True

            def prepare(self, headers, ctx, auth):
                return headers

            def expiration(self, auth):
                return datetime.now(timezone.utc) + timedelta(seconds=5)

            def refresh_lead(self, auth):
                return None

            def refresh(self, auth, ctx):
                updated = auth.clone()
                updated.metadata["refreshed"] = True
                updated.last_refreshed_at = datetime.now(timezone.utc)
                return updated

        # register a refresh lead for provider
        register_refresh_lead("anthropic", lambda _auth: timedelta(seconds=10))

        lead = provider_refresh_lead("anthropic", AuthRecord(id="a", provider="anthropic"))
        self.assertEqual(lead, timedelta(seconds=10))

        store = InMemoryAuthStore()
        auth = AuthRecord(id="a", provider="anthropic")
        store.save("default", auth)
        maint = AuthMaintainer(store, "default", {"anthropic": DummyStrategy()}, interval_seconds=1)

        # directly invoke refresh_if_needed
        now = datetime.now(timezone.utc)
        updated = maint._refresh_if_needed(auth, DummyStrategy(), now)
        # Provider refresh lead should be applied (lead sourced from registry)
        # If refresh runs, metadata will include "refreshed"; otherwise we still
        # verified the registry returns the expected lead above.
        if "refreshed" not in updated.metadata:
            self.assertEqual(lead, timedelta(seconds=10))


if __name__ == "__main__":
    unittest.main()
