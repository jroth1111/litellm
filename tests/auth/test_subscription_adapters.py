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
        url3 = md._models_endpoint(  # type: ignore[attr-defined]
            "https://generativelanguage.googleapis.com", api_version="v1beta"
        )
        self.assertEqual(url3, "https://generativelanguage.googleapis.com/v1beta/models")

    def test_parse_models_payload(self):
        md = _load_module(MODEL_DISCOVERY, "litellm.auth.model_discovery2")
        parsed = md._parse_models_payload(  # type: ignore[attr-defined]
            {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}
        )
        self.assertEqual(parsed, ["gpt-4o", "gpt-4o-mini"])

    def test_register_adapter_paths(self):
        reg = _load_module(
            ADAPTERS_DIR / "registry.py", "litellm.auth.adapters.registry_custom"
        )
        mod_name = "litellm.auth.adapters._custom_test_adapter"
        module = types.ModuleType(mod_name)

        class DummyAdapter:
            provider = "custom"
            capabilities = types.SimpleNamespace(login_flow="device")

        module.DummyAdapter = DummyAdapter
        sys.modules[mod_name] = module

        reg.register_adapter_paths([f"{mod_name}:DummyAdapter"])
        adapter = reg.get_adapter("custom")
        self.assertIsNotNone(adapter)
        self.assertEqual(getattr(adapter, "provider", None), "custom")
        reg.reset_adapter_paths()

    def test_background_model_refresher_registration(self):
        md = _load_module(MODEL_DISCOVERY, "litellm.auth.model_discovery_bg")
        refresher = md.BackgroundModelRefresher(
            cache=md.ModelDiscoveryCache(ttl_seconds=60),
            interval_seconds=30,
        )

        # Test provider registration
        refresher.register_provider(
            provider="test_provider",
            base_url="https://api.example.com",
            auth_records=lambda: [],
            prepare_headers=lambda h, c, a: h,
        )

        # Verify registration
        key = "test_provider:https://api.example.com"
        self.assertIn(key, refresher._providers)
        self.assertEqual(refresher._providers[key]["provider"], "test_provider")

        # Test unregistration
        refresher.unregister_provider("test_provider", "https://api.example.com")
        self.assertNotIn(key, refresher._providers)

    def test_background_model_refresher_start_stop(self):
        md = _load_module(MODEL_DISCOVERY, "litellm.auth.model_discovery_bg2")
        refresher = md.BackgroundModelRefresher(
            cache=md.ModelDiscoveryCache(ttl_seconds=60),
            interval_seconds=30,
        )

        # Test start
        refresher.start()
        self.assertIsNotNone(refresher._thread)
        self.assertTrue(refresher._thread.is_alive())

        # Starting again should be a no-op
        original_thread = refresher._thread
        refresher.start()
        self.assertIs(refresher._thread, original_thread)

        # Test stop
        refresher.stop()
        self.assertFalse(refresher._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
