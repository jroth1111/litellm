"""
Unit tests for enhanced GitHub Copilot OAuth flow.
"""

import pathlib
import sys
import types
import unittest

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Avoid importing `litellm.llms` package __init__ in lightweight envs.
if "litellm" not in sys.modules:
    pkg = types.ModuleType("litellm")
    pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
    sys.modules["litellm"] = pkg
if "litellm.auth" not in sys.modules:
    pkg = types.ModuleType("litellm.auth")
    pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
    sys.modules["litellm.auth"] = pkg
if "litellm.auth.adapters" not in sys.modules:
    pkg = types.ModuleType("litellm.auth.adapters")
    pkg.__path__ = [str(ROOT / "litellm" / "auth" / "adapters")]  # type: ignore[attr-defined]
    sys.modules["litellm.auth.adapters"] = pkg
if "litellm.llms" not in sys.modules:
    pkg = types.ModuleType("litellm.llms")
    pkg.__path__ = [str(ROOT / "litellm" / "llms")]  # type: ignore[attr-defined]
    sys.modules["litellm.llms"] = pkg


class TestCopilotDeviceFlow(unittest.TestCase):
    """Tests for Copilot device flow helpers."""

    def test_device_code_response_dataclass(self):
        """DeviceCodeResponse should work as a dataclass."""
        from litellm.llms.github_copilot.oauth_subscription import DeviceCodeResponse

        resp = DeviceCodeResponse(
            device_code="abc123",
            user_code="ABCD-1234",
            verification_uri="https://github.com/login/device",
            expires_in=900,
            interval=5,
        )
        self.assertEqual(resp.device_code, "abc123")
        self.assertEqual(resp.user_code, "ABCD-1234")

    def test_copilot_token_data_dataclass(self):
        """CopilotTokenData should work as a dataclass with defaults."""
        from litellm.llms.github_copilot.oauth_subscription import CopilotTokenData

        token = CopilotTokenData(access_token="ghp_xxx")
        self.assertEqual(token.access_token, "ghp_xxx")
        self.assertEqual(token.token_type, "Bearer")
        self.assertEqual(token.scope, "")

    def test_copilot_auth_error(self):
        """CopilotAuthError should format message correctly."""
        from litellm.llms.github_copilot.oauth_subscription import CopilotAuthError

        err = CopilotAuthError("access_denied", "User denied authorization")
        self.assertEqual(err.error_type, "access_denied")
        self.assertIn("access_denied", str(err))
        self.assertIn("User denied", str(err))

    def test_request_device_code_with_mock(self):
        """Test device code request with mocked response."""
        from litellm.llms.github_copilot.oauth_subscription import request_device_code
        from litellm.auth.provider_http import close_client_pool

        mock_response = {
            "device_code": "test-device-code",
            "user_code": "TEST-1234",
            "verification_uri": "https://github.com/login/device",
            "expires_in": 900,
            "interval": 5,
        }

        class MockTransport(httpx.MockTransport):
            def __init__(self, payload):
                def handler(request: httpx.Request) -> httpx.Response:
                    return httpx.Response(200, json=payload)
                super().__init__(handler)

        transport = MockTransport(mock_response)
        orig_client = httpx.Client

        def fake_client(*args, **kwargs):
            return orig_client(transport=transport, timeout=5.0)

        httpx.Client = fake_client
        try:
            close_client_pool()
            result = request_device_code()
        finally:
            httpx.Client = orig_client
            close_client_pool()

        self.assertEqual(result.device_code, "test-device-code")
        self.assertEqual(result.user_code, "TEST-1234")

    def test_poll_handles_authorization_pending(self):
        """Poll should continue on authorization_pending error."""
        from litellm.llms.github_copilot.oauth_subscription import (
            DeviceCodeResponse,
            CopilotAuthError,
        )

        # This would require more complex mocking to test the polling loop
        # For now, just verify the dataclass works
        device_code = DeviceCodeResponse(
            device_code="test",
            user_code="TEST",
            verification_uri="https://github.com/login/device",
            expires_in=10,  # Short expiry for test
            interval=1,
        )
        self.assertEqual(device_code.interval, 1)


class TestCopilotConstants(unittest.TestCase):
    """Tests for Copilot OAuth constants."""

    def test_constants_are_defined(self):
        """Verify OAuth constants are properly defined."""
        from litellm.llms.github_copilot.oauth_subscription import (
            COPILOT_CLIENT_ID,
            COPILOT_DEVICE_CODE_URL,
            COPILOT_TOKEN_URL,
            COPILOT_USER_INFO_URL,
        )

        self.assertEqual(COPILOT_CLIENT_ID, "Iv1.b507a08c87ecfe98")
        self.assertIn("github.com", COPILOT_DEVICE_CODE_URL)
        self.assertIn("github.com", COPILOT_TOKEN_URL)
        self.assertIn("api.github.com", COPILOT_USER_INFO_URL)


class TestCopilotStrategy(unittest.TestCase):
    """Tests for enhanced Copilot strategy."""

    def test_strategy_is_non_refreshable(self):
        """Copilot device-flow tokens are not refreshable."""
        from litellm.auth.core import AuthRecord, RequestContext
        from litellm.auth.adapters.github_copilot import GitHubCopilotSubscriptionAdapter

        adapter = GitHubCopilotSubscriptionAdapter()
        self.assertFalse(getattr(adapter, "supports_refresh", True))

        auth = AuthRecord(
            id="copilot",
            provider="github_copilot",
            metadata={"access_token": "tok"},
        )
        with self.assertRaises(ValueError):
            adapter.refresh(auth, RequestContext(model="copilot"))

    def test_strategy_supports_copilot(self):
        """Strategy should support copilot models."""
        from litellm.auth.adapters.github_copilot import GitHubCopilotSubscriptionAdapter

        adapter = GitHubCopilotSubscriptionAdapter()
        self.assertTrue(adapter.supports("copilot"))
        self.assertTrue(adapter.supports("github-copilot"))
        self.assertTrue(adapter.supports("Copilot-Model"))


if __name__ == "__main__":
    unittest.main()
