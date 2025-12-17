import importlib.util
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
AUTH_REG = ROOT / "litellm" / "auth" / "selector_registry.py"
AUTH_CORE = ROOT / "litellm" / "auth" / "core.py"
AUTH_SELECTOR = ROOT / "litellm" / "auth" / "selector.py"


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


class SelectorRegistryTests(unittest.TestCase):
    def test_register_and_get_selector(self):
        if "litellm" not in sys.modules:
            pkg = types.ModuleType("litellm")
            pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
            sys.modules["litellm"] = pkg
        if "litellm.auth" not in sys.modules:
            pkg = types.ModuleType("litellm.auth")
            pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
            sys.modules["litellm.auth"] = pkg

        reg_mod = _load_module(AUTH_REG, "litellm.auth.selector_registry")
        core_mod = _load_module(AUTH_CORE, "litellm.auth.core")
        selector_mod = _load_module(AUTH_SELECTOR, "litellm.auth.selector")

        register_selector = reg_mod.register_selector
        get_selector = reg_mod.get_selector
        CredentialSelector = selector_mod.CredentialSelector
        AuthRecord = core_mod.AuthRecord

        register_selector("default", lambda: CredentialSelector())
        sel = get_selector("default")
        self.assertIsNotNone(sel)
        # ensure selector can operate
        auth = AuthRecord(id="a", provider="anthropic")
        res = sel.select("anthropic/claude", [auth])
        self.assertIsNotNone(res)


if __name__ == "__main__":
    unittest.main()
