"""
Unit tests for Antigravity OAuth flow.
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


class TestAntigravityOAuth(unittest.TestCase):
    """Tests for Antigravity OAuth helpers and strategy."""

    def test_oauth_constants_defined(self):
        """OAuth constants should be properly defined."""
        from litellm.llms.antigravity.antigravity_oauth import (
            ANTIGRAVITY_CLIENT_ID,
            ANTIGRAVITY_CLIENT_SECRET,
            ANTIGRAVITY_AUTH_URL,
            ANTIGRAVITY_TOKEN_URL,
            ANTIGRAVITY_SCOPES,
        )
        
        self.assertIn("accounts.google.com", ANTIGRAVITY_AUTH_URL)
        self.assertIn("oauth2.googleapis.com", ANTIGRAVITY_TOKEN_URL)
        self.assertIsInstance(ANTIGRAVITY_CLIENT_ID, str)
        self.assertTrue(
            ANTIGRAVITY_CLIENT_SECRET is None
            or isinstance(ANTIGRAVITY_CLIENT_SECRET, str)
        )
        self.assertIn("cloud-platform", " ".join(ANTIGRAVITY_SCOPES))

    def test_generate_auth_url(self):
        """Auth URL generation should produce valid URL."""
        from litellm.llms.antigravity.antigravity_oauth import generate_auth_url
        from litellm.auth.pkce import generate_state
        
        state = generate_state()
        url = generate_auth_url(state)
        
        self.assertIn("accounts.google.com/o/oauth2", url)
        self.assertIn("client_id=", url)
        self.assertIn("state=", url)
        self.assertIn("access_type=offline", url)

    def test_token_data_dataclass(self):
        """Token data should work as a dataclass."""
        from litellm.llms.antigravity.antigravity_oauth import AntigravityTokenData
        
        token = AntigravityTokenData(
            access_token="test-token",
            refresh_token="test-refresh",
            project_id="test-project",
        )
        self.assertEqual(token.access_token, "test-token")
        self.assertEqual(token.project_id, "test-project")
        self.assertEqual(token.token_type, "Bearer")

    def test_auth_error(self):
        """Auth error should format message correctly."""
        from litellm.llms.antigravity.antigravity_oauth import AntigravityAuthError
        
        err = AntigravityAuthError("token_exchange_failed", "Invalid code")
        self.assertEqual(err.error_type, "token_exchange_failed")
        self.assertIn("token_exchange_failed", str(err))

    def test_strategy_supports_antigravity(self):
        """Adapter should support antigravity models."""
        from litellm.auth.adapters.antigravity import AntigravitySubscriptionAdapter
        
        adapter = AntigravitySubscriptionAdapter()
        self.assertTrue(adapter.supports("antigravity"))
        self.assertTrue(adapter.supports("antigravity/model"))
        self.assertTrue(adapter.supports("Antigravity-Pro"))


if __name__ == "__main__":
    unittest.main()
