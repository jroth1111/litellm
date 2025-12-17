import importlib.util
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEVICE_FLOW = ROOT / "litellm" / "auth" / "device_flow.py"
TOKEN_STORAGE = ROOT / "litellm" / "auth" / "token_storage.py"


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


class DeviceFlowHelperTests(unittest.TestCase):
    def test_device_flow_poll_authorized(self):
        if "litellm" not in sys.modules:
            pkg = types.ModuleType("litellm")
            pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
            sys.modules["litellm"] = pkg
        if "litellm.auth" not in sys.modules:
            pkg = types.ModuleType("litellm.auth")
            pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
            sys.modules["litellm.auth"] = pkg

        df_mod = _load_module(DEVICE_FLOW, "litellm.auth.device_flow")
        ts_mod = _load_module(TOKEN_STORAGE, "litellm.auth.token_storage")

        start_device_flow = df_mod.start_device_flow
        poll_device_flow = df_mod.poll_device_flow
        mark_device_flow_authorized = df_mod.mark_device_flow_authorized
        InMemoryTokenStorage = ts_mod.InMemoryTokenStorage

        store = InMemoryTokenStorage()
        start_device_flow(
            store,
            token_id="dev1",
            provider="anthropic",
            verification_uri="http://verify",
            user_code="user",
            device_code="device",
            interval_seconds=0,
            expires_in=60,
        )
        mark_device_flow_authorized(store, "dev1", access_token="tok", refresh_token="rt")
        rec = poll_device_flow(store, "dev1", max_attempts=1, sleep_func=lambda _: None)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.metadata.get("access_token"), "tok")


if __name__ == "__main__":
    unittest.main()
