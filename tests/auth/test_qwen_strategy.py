"""
Unit tests for QwenSubscriptionAdapter.
"""

import pathlib
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Avoid importing `litellm.llms`/`litellm.auth` package __init__ in lightweight envs.
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


class TestQwenSubscriptionAdapter(unittest.TestCase):
    """Tests for QwenSubscriptionAdapter."""

    def _get_strategy(self):
        """Import adapter dynamically to avoid import errors."""
        from litellm.auth.adapters.qwen import QwenSubscriptionAdapter
        return QwenSubscriptionAdapter()

    def _get_auth_record(self, **metadata):
        """Create an AuthRecord for testing."""
        from litellm.auth.core import AuthRecord
        return AuthRecord(
            id="test-qwen",
            provider="qwen",
            metadata=metadata,
        )

    def _get_request_context(self):
        """Create a RequestContext for testing."""
        from litellm.auth.core import RequestContext
        return RequestContext(model="qwen/qwen-max")

    def test_provider_name(self):
        """Strategy should have correct provider name."""
        strategy = self._get_strategy()
        self.assertEqual(strategy.provider, "qwen")

    def test_supports_qwen_models(self):
        """Strategy should support qwen models."""
        strategy = self._get_strategy()
        self.assertTrue(strategy.supports("qwen/qwen-max"))
        self.assertTrue(strategy.supports("qwen-turbo"))
        self.assertTrue(strategy.supports("Qwen-Max"))
        self.assertTrue(strategy.supports("dashscope/qwen-plus"))
        self.assertTrue(strategy.supports("some-qwen-model"))

    def test_does_not_support_other_models(self):
        """Strategy should not support non-qwen models."""
        strategy = self._get_strategy()
        self.assertFalse(strategy.supports("gpt-4"))
        self.assertFalse(strategy.supports("claude-3"))
        self.assertFalse(strategy.supports("gemini-pro"))

    def test_prepare_adds_bearer_token(self):
        """Prepare should add Authorization header with Bearer token."""
        strategy = self._get_strategy()
        auth = self._get_auth_record(access_token="test-token-123")
        ctx = self._get_request_context()

        headers = strategy.prepare({}, ctx, auth)
        self.assertEqual(headers["Authorization"], "Bearer test-token-123")

    def test_prepare_raises_without_token(self):
        """Prepare should raise ValueError if no access_token."""
        strategy = self._get_strategy()
        auth = self._get_auth_record()
        ctx = self._get_request_context()

        with self.assertRaises(ValueError) as cm:
            strategy.prepare({}, ctx, auth)
        self.assertIn("missing access_token", str(cm.exception))

    def test_expiration_parses_iso_format(self):
        """Expiration should parse ISO format expires_at."""
        strategy = self._get_strategy()
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        auth = self._get_auth_record(expires_at=future.isoformat())

        exp = strategy.expiration(auth)
        self.assertIsNotNone(exp)
        # Should be close to the expected time
        self.assertAlmostEqual(exp.timestamp(), future.timestamp(), delta=1)

    def test_expiration_returns_none_for_missing(self):
        """Expiration should return None if no expiry in metadata."""
        strategy = self._get_strategy()
        auth = self._get_auth_record()

        exp = strategy.expiration(auth)
        self.assertIsNone(exp)

    def test_refresh_lead_returns_default(self):
        """Refresh lead should return default of 5 minutes."""
        strategy = self._get_strategy()
        auth = self._get_auth_record()

        lead = strategy.refresh_lead(auth)
        self.assertEqual(lead, timedelta(minutes=5))

    def test_refresh_with_mocked_endpoint(self):
        """Test token refresh with mocked HTTP response."""
        strategy = self._get_strategy()
        auth = self._get_auth_record(refresh_token="rt-old")
        auth.attributes["token_url"] = "https://mock.qwen.ai/token"
        ctx = self._get_request_context()

        payload = {
            "access_token": "new-token",
            "refresh_token": "new-rt",
            "expires_in": 3600,
        }

        class MockTransport(httpx.MockTransport):
            def __init__(self, resp_payload):
                def handler(request: httpx.Request) -> httpx.Response:
                    return httpx.Response(200, json=resp_payload)
                super().__init__(handler)

        transport = MockTransport(payload)
        orig_client = httpx.Client

        def fake_client(*args, **kwargs):
            return orig_client(transport=transport, timeout=5.0)

        httpx.Client = fake_client
        try:
            refreshed = strategy.refresh(auth, ctx)
        finally:
            httpx.Client = orig_client

        self.assertEqual(refreshed.metadata["access_token"], "new-token")
        self.assertEqual(refreshed.metadata["refresh_token"], "new-rt")
        self.assertIn("expires_at", refreshed.metadata)

    def test_refresh_raises_without_refresh_token(self):
        """Refresh should raise ValueError if no refresh_token."""
        strategy = self._get_strategy()
        auth = self._get_auth_record()
        ctx = self._get_request_context()

        with self.assertRaises(ValueError) as cm:
            strategy.refresh(auth, ctx)
        self.assertIn("missing refresh_token", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
