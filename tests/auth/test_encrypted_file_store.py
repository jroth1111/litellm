import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest

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


class EncryptedFileStoreTests(unittest.TestCase):
    def test_encrypted_store_roundtrip_and_envelope_shape(self):
        # Minimal package scaffolding to avoid importing litellm/__init__.py
        if "litellm" not in sys.modules:
            pkg = types.ModuleType("litellm")
            pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
            sys.modules["litellm"] = pkg
        if "litellm.auth" not in sys.modules:
            pkg = types.ModuleType("litellm.auth")
            pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
            sys.modules["litellm.auth"] = pkg

        core_mod = _load_module(AUTH_CORE, "litellm.auth.core")
        store_mod = _load_module(AUTH_FILE_STORE, "litellm.auth.file_store")

        AuthRecord = core_mod.AuthRecord
        EncryptedJsonFileAuthStore = store_mod.EncryptedJsonFileAuthStore

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                store = EncryptedJsonFileAuthStore(tmpdir, secret="unit-test-secret")
            except Exception as e:
                self.skipTest(f"encryption backend not available: {e}")
            rec = AuthRecord(
                id="r1",
                provider="anthropic",
                label="acct",
                metadata={"access_token": "at", "refresh_token": "rt"},
            )
            store.save("default", rec)

            # File on disk should be an envelope containing ciphertext, not plaintext record fields.
            path = pathlib.Path(tmpdir) / "default" / "r1.json"
            raw = path.read_text(encoding="utf-8")
            env = json.loads(raw)
            self.assertIn("ct", env)
            self.assertIn("alg", env)
            self.assertNotIn("metadata", env)

            loaded = store.get("default", "r1")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.metadata.get("access_token"), "at")
            self.assertEqual(loaded.metadata.get("refresh_token"), "rt")

    def test_plaintext_fallback_allows_migration(self):
        # Minimal package scaffolding to avoid importing litellm/__init__.py
        if "litellm" not in sys.modules:
            pkg = types.ModuleType("litellm")
            pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
            sys.modules["litellm"] = pkg
        if "litellm.auth" not in sys.modules:
            pkg = types.ModuleType("litellm.auth")
            pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
            sys.modules["litellm.auth"] = pkg

        core_mod = _load_module(AUTH_CORE, "litellm.auth.core_mig")
        store_mod = _load_module(AUTH_FILE_STORE, "litellm.auth.file_store_mig")

        AuthRecord = core_mod.AuthRecord
        JsonFileAuthStore = store_mod.JsonFileAuthStore
        EncryptedJsonFileAuthStore = store_mod.EncryptedJsonFileAuthStore

        with tempfile.TemporaryDirectory() as tmpdir:
            plain = JsonFileAuthStore(tmpdir)
            rec = AuthRecord(
                id="r2",
                provider="openai",
                label="acct",
                metadata={"access_token": "at2"},
            )
            plain.save("default", rec)

            try:
                enc = EncryptedJsonFileAuthStore(
                    tmpdir,
                    secret="unit-test-secret",
                    allow_plaintext_fallback=True,
                )
            except Exception as e:
                self.skipTest(f"encryption backend not available: {e}")
            loaded = enc.get("default", "r2")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.metadata.get("access_token"), "at2")


if __name__ == "__main__":
    unittest.main()
