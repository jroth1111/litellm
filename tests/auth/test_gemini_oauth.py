"""
Unit tests for Gemini OAuth flow.
"""

import pathlib
import sys
import types
import unittest

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


class TestGeminiOAuth(unittest.TestCase):
    """Tests for Gemini OAuth helpers and strategy."""

    def test_oauth_constants_defined(self):
        """OAuth constants should be properly defined."""
        from litellm.llms.gemini.oauth_subscription import (
            GEMINI_CLIENT_ID,
            GEMINI_CLIENT_SECRET,
            GEMINI_AUTHORIZE_URL,
            GEMINI_TOKEN_URL,
            GEMINI_SCOPES,
        )
        
        self.assertIn("accounts.google.com", GEMINI_AUTHORIZE_URL)
        self.assertIn("oauth2.googleapis.com", GEMINI_TOKEN_URL)
        self.assertIsInstance(GEMINI_CLIENT_ID, str)
        self.assertTrue(
            GEMINI_CLIENT_SECRET is None or isinstance(GEMINI_CLIENT_SECRET, str)
        )
        self.assertIn("cloud-platform", " ".join(GEMINI_SCOPES))

    def test_generate_auth_url(self):
        """Auth URL generation should produce valid URL."""
        from litellm.llms.gemini.oauth_subscription import generate_auth_url
        from litellm.auth.pkce import generate_state
        
        state = generate_state()
        url = generate_auth_url(state)
        
        self.assertIn("accounts.google.com/o/oauth2", url)
        self.assertIn("client_id=", url)
        self.assertIn("state=", url)
        self.assertIn("redirect_uri=", url)

    def test_token_data_dataclass(self):
        """Token data should work as a dataclass."""
        from litellm.llms.gemini.oauth_subscription import GeminiTokenData
        
        token = GeminiTokenData(
            access_token="test-token",
            refresh_token="test-refresh",
            email="test@example.com",
        )
        self.assertEqual(token.access_token, "test-token")
        self.assertEqual(token.email, "test@example.com")
        self.assertEqual(token.token_type, "Bearer")

    def test_auth_error(self):
        """Auth error should format message correctly."""
        from litellm.llms.gemini.oauth_subscription import GeminiAuthError
        
        err = GeminiAuthError("token_exchange_failed", "Invalid code")
        self.assertEqual(err.error_type, "token_exchange_failed")
        self.assertIn("token_exchange_failed", str(err))

    def test_strategy_supports_gemini(self):
        """Adapter should support gemini models."""
        from litellm.auth.adapters.gemini import GeminiSubscriptionAdapter
        
        adapter = GeminiSubscriptionAdapter()
        self.assertTrue(adapter.supports("gemini-pro"))
        self.assertTrue(adapter.supports("gemini/gemini-1.5-pro"))
        self.assertTrue(adapter.supports("google/gemini-ultra"))


if __name__ == "__main__":
    unittest.main()
