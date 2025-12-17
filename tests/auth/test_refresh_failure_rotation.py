import importlib.util
import pathlib
import sys
import types
import unittest
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]

AUTH_CORE = ROOT / "litellm" / "auth" / "core.py"
AUTH_SELECTOR = ROOT / "litellm" / "auth" / "selector.py"
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

core_mod = _load_module(AUTH_CORE, "litellm.auth.core")
selector_mod = _load_module(AUTH_SELECTOR, "litellm.auth.selector")
mem_store_mod = _load_module(AUTH_MEM_STORE, "litellm.auth.memory_store")

AuthRecord = core_mod.AuthRecord
AuthStatus = core_mod.AuthStatus
CredentialSelector = selector_mod.CredentialSelector
InMemoryAuthStore = mem_store_mod.InMemoryAuthStore


class FakeHttpError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class MiniRotateProxy:
    """
    Minimal same-request rotation loop: on 401 refresh failure, mark auth as unavailable and retry with next auth.
    """

    def __init__(self, selector: CredentialSelector, store: InMemoryAuthStore):
        self.selector = selector
        self.store = store

    def call(
        self,
        logical_model: str,
        auth_records: list,
        now: datetime,
        namespace: str = "default",
    ):
        attempted: set[str] = set()
        refreshed: set[str] = set()
        max_attempts = len(auth_records)
        attempts = 0

        while attempts < max_attempts:
            candidates = [
                rec for rec in auth_records if getattr(rec, "id", None) not in attempted
            ]
            sel = self.selector.select(logical_model=logical_model, auth_records=candidates, now=now)
            if sel is None:
                break

            try:
                return self._do_call(sel.auth)
            except FakeHttpError as exc:
                if exc.status_code in (401, 403) and sel.auth.id not in refreshed:
                    refreshed.add(sel.auth.id)
                    try:
                        self._refresh(sel.auth)
                    except Exception:
                        # Refresh failed; fall through to rotation for this auth.
                        pass

                if exc.status_code not in (401, 403, 429):
                    raise

                updated = self.selector.mark_failure(
                    sel.auth,
                    provider_model=sel.provider_model,
                    error_message=str(exc),
                    status_code=exc.status_code,
                    is_quota=(exc.status_code == 429),
                    store=self.store,
                    namespace=namespace,
                    now=now,
                )
                for i, rec in enumerate(auth_records):
                    if rec.id == updated.id:
                        auth_records[i] = updated
                        break
                attempted.add(sel.auth.id)
                attempts += 1
                continue

        raise FakeHttpError(401)

    def _do_call(self, auth: AuthRecord):
        token = auth.metadata.get("access_token")
        if token and str(token).startswith("invalid"):
            raise FakeHttpError(401)
        return "ok"

    def _refresh(self, auth: AuthRecord):
        raise ValueError("missing refresh_token")


class RefreshFailureRotationTests(unittest.TestCase):
    def test_rotates_to_next_auth_when_refresh_fails(self):
        selector = CredentialSelector()
        store = InMemoryAuthStore()

        auth1 = AuthRecord(
            id="a1",
            provider="anthropic",
            metadata={"access_token": "invalid-1"},
            attributes={},
            status=AuthStatus.ACTIVE,
        )
        auth2 = AuthRecord(
            id="a2",
            provider="anthropic",
            metadata={"access_token": "ok"},
            attributes={},
            status=AuthStatus.ACTIVE,
        )
        store.save("default", auth1)
        store.save("default", auth2)

        now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        proxy = MiniRotateProxy(selector, store)
        auth_records = [auth1, auth2]

        result = proxy.call("anthropic/claude", auth_records=auth_records, now=now)
        self.assertEqual(result, "ok")

        updated_auth1 = store.get("default", "a1")
        self.assertIsNotNone(updated_auth1)
        self.assertTrue(updated_auth1.unavailable)
        self.assertEqual(updated_auth1.status, AuthStatus.ERROR)
        self.assertIsNotNone(updated_auth1.next_retry_after)


if __name__ == "__main__":
    unittest.main()
