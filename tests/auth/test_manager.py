import importlib.util
import pathlib
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUTH_CORE = ROOT / "litellm" / "auth" / "core.py"
AUTH_MANAGER = ROOT / "litellm" / "auth" / "manager.py"
AUTH_MEM = ROOT / "litellm" / "auth" / "memory_store.py"


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


class DummyStrategy:
    provider = "anthropic"

    def supports(self, model):
        return True

    def prepare(self, headers, ctx, auth):
        return headers

    def expiration(self, auth):
        return datetime.now(timezone.utc) + timedelta(seconds=1)

    def refresh_lead(self, auth):
        return timedelta(seconds=5)

    def refresh(self, auth, ctx):
        updated = auth.clone()
        updated.metadata["refreshed"] = True
        updated.runtime = auth.runtime
        return updated


class ManagerTests(unittest.TestCase):
    def test_refresh_preserves_runtime_and_saves(self):
        # scaffold package paths
        if "litellm" not in sys.modules:
            pkg = types.ModuleType("litellm")
            pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
            sys.modules["litellm"] = pkg
        if "litellm.auth" not in sys.modules:
            pkg = types.ModuleType("litellm.auth")
            pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
            sys.modules["litellm.auth"] = pkg

        core_mod = _load_module(AUTH_CORE, "litellm.auth.core")
        manager_mod = _load_module(AUTH_MANAGER, "litellm.auth.manager")
        mem_mod = _load_module(AUTH_MEM, "litellm.auth.memory_store")

        AuthRecord = core_mod.AuthRecord
        InMemoryAuthStore = mem_mod.InMemoryAuthStore
        AuthManager = manager_mod.AuthManager

        store = InMemoryAuthStore()
        auth = AuthRecord(id="a", provider="anthropic", metadata={}, runtime={"session": "x"})
        store.save("default", auth)
        mgr = AuthManager(store, strategies={"anthropic": DummyStrategy()}, namespace="default")

        refreshed = mgr.refresh_auth(auth, DummyStrategy())
        self.assertEqual(refreshed.metadata.get("refreshed"), True)
        self.assertEqual(refreshed.runtime, {"session": "x"})
        persisted = store.get("default", "a")
        self.assertEqual(persisted.metadata.get("refreshed"), True)

    def test_mark_failure_sets_retry_after(self):
        core_mod = _load_module(AUTH_CORE, "litellm.auth.core")
        manager_mod = _load_module(AUTH_MANAGER, "litellm.auth.manager")
        mem_mod = _load_module(AUTH_MEM, "litellm.auth.memory_store")
        AuthRecord = core_mod.AuthRecord
        InMemoryAuthStore = mem_mod.InMemoryAuthStore
        AuthManager = manager_mod.AuthManager

        store = InMemoryAuthStore()
        auth = AuthRecord(id="a", provider="anthropic")
        store.save("default", auth)
        mgr = AuthManager(store, strategies={"anthropic": DummyStrategy()}, namespace="default")

        class Err(Exception):
            def __init__(self):
                super().__init__("boom")
                self.status_code = 429
                self.headers = {"retry-after": "120"}

        updated = mgr.mark_result(auth, "anthropic/claude", success=False, error=Err())
        self.assertIsNotNone(updated.next_retry_after)
        self.assertGreaterEqual(updated.next_retry_after, datetime.now(timezone.utc) + timedelta(seconds=119))

    def test_refresh_evaluator_blocks_refresh(self):
        core_mod = _load_module(AUTH_CORE, "litellm.auth.core")
        manager_mod = _load_module(AUTH_MANAGER, "litellm.auth.manager")
        mem_mod = _load_module(AUTH_MEM, "litellm.auth.memory_store")
        AuthRecord = core_mod.AuthRecord
        InMemoryAuthStore = mem_mod.InMemoryAuthStore
        AuthManager = manager_mod.AuthManager

        blocked = []

        def eval_block(record, strat, now):
            blocked.append(record.id)
            return False

        store = InMemoryAuthStore()
        auth = AuthRecord(id="a", provider="anthropic")
        store.save("default", auth)
        mgr = AuthManager(
            store,
            strategies={"anthropic": DummyStrategy()},
            namespace="default",
            refresh_evaluators={"anthropic": eval_block},
            refresh_backoffs={"anthropic": {"failure": 123, "pending": 45}},
        )
        refreshed = mgr.refresh_if_due(auth)
        self.assertNotIn("refreshed", refreshed.metadata)
        self.assertEqual(blocked, ["a"])


if __name__ == "__main__":
    unittest.main()
