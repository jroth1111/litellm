import importlib.util
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]

ADAPTERS_DIR = ROOT / "litellm" / "auth" / "adapters"
MODEL_DISCOVERY = ROOT / "litellm" / "auth" / "model_discovery.py"


def _ensure_pkg(name: str, path: pathlib.Path) -> None:
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _pop_module_tree(prefix: str) -> dict:
    """
    Remove a module and all submodules from sys.modules.

    Returns a dict of removed modules for restoration.
    """
    removed = {}
    keys = list(sys.modules.keys())
    for key in keys:
        if key == prefix or key.startswith(prefix + "."):
            removed[key] = sys.modules.pop(key)
    return removed


class SubscriptionAdaptersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Avoid importing `litellm/__init__.py` in lightweight envs.
        _ensure_pkg("litellm", ROOT / "litellm")
        _ensure_pkg("litellm.auth", ROOT / "litellm" / "auth")
        _ensure_pkg("litellm.auth.adapters", ADAPTERS_DIR)

    def test_registry_get_adapter(self):
        reg = _load_module(
            ADAPTERS_DIR / "registry.py", "litellm.auth.adapters.registry"
        )
        adapter = reg.get_adapter("openai")
        self.assertIsNotNone(adapter)
        self.assertEqual(getattr(adapter, "provider", None), "openai")
        caps = getattr(adapter, "capabilities", None)
        self.assertIsNotNone(caps)
        self.assertEqual(getattr(caps, "login_flow", None), "browser_pkce")

    def test_oauth_loader_does_not_import_litellm_llms(self):
        reg = _load_module(
            ADAPTERS_DIR / "registry.py", "litellm.auth.adapters.registry2"
        )
        adapter = reg.get_adapter("openai")
        self.assertIsNotNone(adapter)

        # Ensure the heavy `litellm.llms` package isn't imported as a side-effect.
        removed = _pop_module_tree("litellm.llms")
        try:
            oauth_mod = adapter._oauth()  # type: ignore[attr-defined]
            self.assertIsNotNone(oauth_mod)
            self.assertNotIn("litellm.llms", sys.modules)
        finally:
            sys.modules.update(removed)

    def test_model_discovery_endpoint_builder(self):
        md = _load_module(MODEL_DISCOVERY, "litellm.auth.model_discovery")
        url1 = md._models_endpoint("https://api.openai.com/v1")  # type: ignore[attr-defined]
        self.assertEqual(url1, "https://api.openai.com/v1/models")
        url2 = md._models_endpoint("https://api.openai.com")  # type: ignore[attr-defined]
        self.assertEqual(url2, "https://api.openai.com/v1/models")

    def test_parse_models_payload(self):
        md = _load_module(MODEL_DISCOVERY, "litellm.auth.model_discovery2")
        parsed = md._parse_models_payload(  # type: ignore[attr-defined]
            {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}
        )
        self.assertEqual(parsed, ["gpt-4o", "gpt-4o-mini"])


if __name__ == "__main__":
    unittest.main()
