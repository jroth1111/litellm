import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOKEN_STORAGE = ROOT / "litellm" / "auth" / "token_storage.py"
TOKEN_MGMT = ROOT / "litellm" / "auth" / "token_management.py"
TEST_AUTH_KEY = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


class TokenManagementTests(unittest.TestCase):
    def test_export_import_tokens(self):
        if "litellm" not in sys.modules:
            pkg = types.ModuleType("litellm")
            pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
            sys.modules["litellm"] = pkg
        if "litellm.auth" not in sys.modules:
            pkg = types.ModuleType("litellm.auth")
            pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
            sys.modules["litellm.auth"] = pkg

        ts_mod = _load_module(TOKEN_STORAGE, "litellm.auth.token_storage")
        mgmt_mod = _load_module(TOKEN_MGMT, "litellm.auth.token_management")
        TokenRecord = ts_mod.TokenRecord
        JsonTokenStorage = ts_mod.JsonTokenStorage
        export_tokens = mgmt_mod.export_tokens
        import_tokens = mgmt_mod.import_tokens

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["LITELLM_AUTH_ENCRYPTION_KEY"] = TEST_AUTH_KEY
            store = JsonTokenStorage(tmpdir, namespace="ns1")
            rec = TokenRecord(
                id="tok1",
                provider="openai",
                scopes=["scopeA"],
                metadata={"code": "xyz"},
            )
            store.save(rec)
            out_path = pathlib.Path(tmpdir) / "tokens.json"
            export_tokens(store, str(out_path))
            self.assertTrue(out_path.exists())
            store2 = JsonTokenStorage(tmpdir, namespace="ns2")
            imported = import_tokens(store2, str(out_path))
            self.assertEqual(len(imported), 1)
            self.assertEqual(store2.list()[0].provider, "openai")
            self.assertEqual(store2.list()[0].scopes, ["scopeA"])


if __name__ == "__main__":
    unittest.main()
