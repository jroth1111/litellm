"""
Unit tests for Cursor AI OAuth device-link polling flow.

These tests load the module directly to avoid requiring full LiteLLM deps.
"""

import importlib.util
import pathlib
import sys
import types
import unittest

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[2]
CURSOR_OAUTH = ROOT / "litellm" / "llms" / "cursor" / "oauth_subscription.py"


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# scaffold minimal package paths for relative imports
if "litellm" not in sys.modules:
    pkg = types.ModuleType("litellm")
    pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
    sys.modules["litellm"] = pkg
if "litellm.llms" not in sys.modules:
    pkg = types.ModuleType("litellm.llms")
    pkg.__path__ = [str(ROOT / "litellm" / "llms")]  # type: ignore[attr-defined]
    sys.modules["litellm.llms"] = pkg
if "litellm.llms.cursor" not in sys.modules:
    pkg = types.ModuleType("litellm.llms.cursor")
    pkg.__path__ = [str(ROOT / "litellm" / "llms" / "cursor")]  # type: ignore[attr-defined]
    sys.modules["litellm.llms.cursor"] = pkg

cursor_mod = _load_module(CURSOR_OAUTH, "litellm.llms.cursor.oauth_subscription")


class TestCursorOAuth(unittest.TestCase):
    def test_oauth_constants_defined(self):
        self.assertIn("cursor.sh", cursor_mod.CURSOR_AUTHENTICATOR_URL)
        self.assertIn("cursor.sh", cursor_mod.CURSOR_API_URL)
        self.assertTrue(cursor_mod.CURSOR_LOGIN_URL.endswith("/login"))
        self.assertIn("/auth/poll", cursor_mod.CURSOR_POLL_URL)
        self.assertIn("/auth/refresh", cursor_mod.CURSOR_REFRESH_URL)

    def test_start_cursor_login_generates_url(self):
        session = cursor_mod.start_cursor_login(verifier_bytes=4)
        self.assertIn("uuid=", session.login_url)
        self.assertIn("verifier=", session.login_url)
        self.assertEqual(
            session.uuid, session.login_url.split("uuid=")[1].split("&")[0]
        )

    def test_poll_for_token_with_mock(self):
        mock_payload = {"accessToken": "at", "refreshToken": "rt"}

        class MockTransport(httpx.MockTransport):
            def __init__(self, payload):
                def handler(_request: httpx.Request) -> httpx.Response:
                    return httpx.Response(200, json=payload)

                super().__init__(handler)

        transport = MockTransport(mock_payload)
        orig_client = httpx.Client

        def fake_client(*args, **kwargs):
            return orig_client(transport=transport, timeout=5.0)

        httpx.Client = fake_client  # type: ignore
        try:
            session = cursor_mod.CursorLoginSession(uuid="u", verifier="v", login_url="x")
            token = cursor_mod.poll_for_token(
                session, max_attempts=1, interval_seconds=0
            )
        finally:
            httpx.Client = orig_client  # type: ignore

        self.assertEqual(token.access_token, "at")
        self.assertEqual(token.refresh_token, "rt")

    def test_refresh_tokens_with_mock(self):
        mock_payload = {
            "accessToken": "new-at",
            "refreshToken": "new-rt",
            "expiresIn": 60,
        }

        class MockTransport(httpx.MockTransport):
            def __init__(self, payload):
                def handler(_request: httpx.Request) -> httpx.Response:
                    return httpx.Response(200, json=payload)

                super().__init__(handler)

        transport = MockTransport(mock_payload)
        orig_client = httpx.Client

        def fake_client(*args, **kwargs):
            return orig_client(transport=transport, timeout=5.0)

        httpx.Client = fake_client  # type: ignore
        try:
            token = cursor_mod.refresh_tokens("rt-old")
        finally:
            httpx.Client = orig_client  # type: ignore

        self.assertEqual(token.access_token, "new-at")
        self.assertEqual(token.refresh_token, "new-rt")


if __name__ == "__main__":
    unittest.main()
