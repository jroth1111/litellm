import importlib.util
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUTH_CORE = ROOT / "litellm" / "auth" / "core.py"


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


class TransportOverrideTests(unittest.TestCase):
    def test_transport_overrides_prefers_attributes(self):
        if "litellm" not in sys.modules:
            pkg = types.ModuleType("litellm")
            pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
            sys.modules["litellm"] = pkg
        if "litellm.auth" not in sys.modules:
            pkg = types.ModuleType("litellm.auth")
            pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
            sys.modules["litellm.auth"] = pkg
        core_mod = _load_module(AUTH_CORE, "litellm.auth.core")
        AuthRecord = core_mod.AuthRecord

        auth = AuthRecord(
            id="a",
            provider="anthropic",
            attributes={"proxy": "http://proxy:1", "mtls_cert": "/c", "mtls_key": "/k"},
            metadata={"proxy": "http://ignored"},
        )
        overrides = auth.transport_overrides()
        self.assertEqual(overrides["proxy"], "http://proxy:1")
        self.assertEqual(overrides["cert"], "/c")
        self.assertEqual(overrides["key"], "/k")

    def test_transport_factory_key(self):
        if "litellm" not in sys.modules:
            pkg = types.ModuleType("litellm")
            pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
            sys.modules["litellm"] = pkg
        if "litellm.auth" not in sys.modules:
            pkg = types.ModuleType("litellm.auth")
            pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
            sys.modules["litellm.auth"] = pkg
        core_mod = _load_module(AUTH_CORE, "litellm.auth.core")
        AuthRecord = core_mod.AuthRecord

        auth = AuthRecord(
            id="b",
            provider="openai",
            metadata={"transport_factory_key": "tenant-1"},
        )
        overrides = auth.transport_overrides()
        self.assertEqual(overrides.get("transport_factory_key"), "tenant-1")


if __name__ == "__main__":
    unittest.main()
