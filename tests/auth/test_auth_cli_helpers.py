import importlib.util
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUTH_CLI = ROOT / "litellm" / "auth" / "cli.py"


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


cli_mod = _load_module(AUTH_CLI, "litellm.auth.cli")


class AuthCliHelpersTests(unittest.TestCase):
    def test_parse_redirect_uri_localhost(self):
        host, port, path = cli_mod._parse_redirect_uri("http://localhost:1234/callback")  # type: ignore[attr-defined]
        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, 1234)
        self.assertEqual(path, "/callback")

    def test_default_auth_id_prefix(self):
        auth_id = cli_mod._default_auth_id("anthropic")  # type: ignore[attr-defined]
        self.assertTrue(auth_id.startswith("anthropic-"))


if __name__ == "__main__":
    unittest.main()
