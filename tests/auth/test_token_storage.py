import importlib.util
import os
import pathlib
import sys
import tempfile
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOKEN_STORAGE_PATH = ROOT / "litellm" / "auth" / "token_storage.py"
TEST_AUTH_KEY = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


class TokenStorageTests(unittest.TestCase):
    def test_save_get_list_delete_file(self):
        # minimal package scaffolding
        if "litellm" not in sys.modules:
            pkg = types.ModuleType("litellm")
            pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
            sys.modules["litellm"] = pkg
        if "litellm.auth" not in sys.modules:
            pkg = types.ModuleType("litellm.auth")
            pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
            sys.modules["litellm.auth"] = pkg

        mod = _load_module(TOKEN_STORAGE_PATH, "litellm.auth.token_storage")
        TokenRecord = mod.TokenRecord
        JsonTokenStorage = mod.JsonTokenStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["LITELLM_AUTH_ENCRYPTION_KEY"] = TEST_AUTH_KEY
            store = JsonTokenStorage(tmpdir)
            rec = TokenRecord(id="t1", provider="anthropic", metadata={"code": "abc"})
            store.save(rec)

            loaded = store.get("t1")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.metadata["code"], "abc")

            all_tokens = store.list()
            self.assertEqual(len(all_tokens), 1)

            store.delete("t1")
            self.assertIsNone(store.get("t1"))

    def test_save_get_list_delete_memory(self):
        # minimal package scaffolding
        if "litellm" not in sys.modules:
            pkg = types.ModuleType("litellm")
            pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
            sys.modules["litellm"] = pkg
        if "litellm.auth" not in sys.modules:
            pkg = types.ModuleType("litellm.auth")
            pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
            sys.modules["litellm.auth"] = pkg

        mod = _load_module(TOKEN_STORAGE_PATH, "litellm.auth.token_storage")
        TokenRecord = mod.TokenRecord
        InMemoryTokenStorage = mod.InMemoryTokenStorage

        store = InMemoryTokenStorage()
        rec = TokenRecord(
            id="t2",
            provider="openai",
            scopes=["scope1"],
            expires_at=None,
            metadata={"code": "def"},
        )
        store.save(rec)
        loaded = store.get("t2")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.metadata["code"], "def")
        self.assertEqual(loaded.scopes, ["scope1"])
        all_tokens = store.list()
        self.assertEqual(len(all_tokens), 1)
        store.delete("t2")
        self.assertIsNone(store.get("t2"))


if __name__ == "__main__":
    unittest.main()
