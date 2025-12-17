import importlib.util
import pathlib
import sys
import types
import unittest

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


class HookRecorder:
    def __init__(self):
        self.calls = []

    def on_success(self, auth, provider_model=None, retry_after=None, is_quota=False):
        self.calls.append(("success", provider_model, retry_after, is_quota))

    def on_failure(
        self,
        auth,
        provider_model=None,
        error=None,
        retry_after=None,
        is_quota=False,
        status_code=None,
    ):
        self.calls.append(("failure", provider_model, retry_after, is_quota, status_code))


class HookContextTests(unittest.TestCase):
    def test_failure_hook_receives_retry_after_and_status(self):
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

        class FakeError(Exception):
            def __init__(self):
                super().__init__("boom")
                self.status_code = 429
                self.headers = {"retry-after": "5"}

        class DummyStrategy:
            provider = "anthropic"

            def supports(self, model):
                return True

            def prepare(self, headers, ctx, auth):
                return headers

            def expiration(self, auth):
                return None

            def refresh_lead(self, auth):
                return None

            def refresh(self, auth, ctx):
                return auth

        store = InMemoryAuthStore()
        auth = AuthRecord(id="a", provider="anthropic")
        store.save("default", auth)
        hooks = HookRecorder()
        mgr = AuthManager(
            store,
            strategies={"anthropic": DummyStrategy()},
            namespace="default",
            hooks=hooks,
        )

        mgr.mark_result(auth, "anthropic/claude", success=False, error=FakeError())
        self.assertEqual(len(hooks.calls), 1)
        kind, provider_model, retry_after, is_quota, status_code = hooks.calls[0]
        self.assertEqual(kind, "failure")
        self.assertEqual(status_code, 429)
        self.assertTrue(is_quota)
        self.assertIsNotNone(retry_after)


if __name__ == "__main__":
    unittest.main()
