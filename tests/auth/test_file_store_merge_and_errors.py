import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUTH_CORE = ROOT / "litellm" / "auth" / "core.py"
AUTH_FILE_STORE = ROOT / "litellm" / "auth" / "file_store.py"


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


class FileStoreMergeAndErrorsTests(unittest.TestCase):
    def test_save_merges_model_states_and_preserves_newer_tokens(self):
        # Minimal package scaffolding to avoid importing litellm/__init__.py
        if "litellm" not in sys.modules:
            pkg = types.ModuleType("litellm")
            pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
            sys.modules["litellm"] = pkg
        if "litellm.auth" not in sys.modules:
            pkg = types.ModuleType("litellm.auth")
            pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
            sys.modules["litellm.auth"] = pkg

        core_mod = _load_module(AUTH_CORE, "litellm.auth.core_merge")
        store_mod = _load_module(AUTH_FILE_STORE, "litellm.auth.file_store_merge")

        AuthRecord = core_mod.AuthRecord
        AuthStatus = core_mod.AuthStatus
        ModelState = core_mod.ModelState
        JsonFileAuthStore = store_mod.JsonFileAuthStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonFileAuthStore(tmpdir)

            t1 = datetime.now(timezone.utc) - timedelta(minutes=10)
            t2 = datetime.now(timezone.utc) - timedelta(minutes=5)

            # Initial record (old token).
            base = AuthRecord(
                id="a",
                provider="openai",
                metadata={"access_token": "old", "refresh_token": "rt"},
                last_refreshed_at=t1,
            )
            store.save("default", base)

            # Simulate another writer refreshing tokens.
            refreshed = base.clone()
            refreshed.metadata["access_token"] = "new"
            refreshed.last_refreshed_at = t2
            store.save("default", refreshed)

            # Simulate a stale writer (still holding old token) updating model state.
            stale = base.clone()
            stale.model_states["openai/gpt-4o"] = ModelState(
                status=AuthStatus.ERROR,
                status_message="rate limit",
                unavailable=True,
            )
            store.save("default", stale)

            loaded = store.get("default", "a")
            self.assertIsNotNone(loaded)
            assert loaded is not None
            # Token should remain the newer one.
            self.assertEqual(loaded.metadata.get("access_token"), "new")
            # Model state update should be present.
            self.assertIn("openai/gpt-4o", loaded.model_states)

    def test_list_records_load_errors_for_malformed_files(self):
        # Minimal package scaffolding to avoid importing litellm/__init__.py
        if "litellm" not in sys.modules:
            pkg = types.ModuleType("litellm")
            pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
            sys.modules["litellm"] = pkg
        if "litellm.auth" not in sys.modules:
            pkg = types.ModuleType("litellm.auth")
            pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
            sys.modules["litellm.auth"] = pkg

        store_mod = _load_module(AUTH_FILE_STORE, "litellm.auth.file_store_err")
        JsonFileAuthStore = store_mod.JsonFileAuthStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonFileAuthStore(tmpdir)
            ns_dir = pathlib.Path(tmpdir) / "default"
            ns_dir.mkdir(parents=True, exist_ok=True)
            (ns_dir / "bad.json").write_text("{not valid json", encoding="utf-8")

            recs = store.list("default")
            self.assertEqual(recs, [])

            errs = getattr(store, "last_load_errors", None) or []
            self.assertEqual(len(errs), 1)
            err0 = errs[0]
            self.assertIn("bad.json", getattr(err0, "path", ""))
            self.assertTrue(getattr(err0, "message", ""))


if __name__ == "__main__":
    unittest.main()

