"""
Unit tests for Anthropic OAuth flow.
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
if "litellm.llms.anthropic" not in sys.modules:
    pkg = types.ModuleType("litellm.llms.anthropic")
    pkg.__path__ = [str(ROOT / "litellm" / "llms" / "anthropic")]  # type: ignore[attr-defined]
    sys.modules["litellm.llms.anthropic"] = pkg


class TestAnthropicOAuth(unittest.TestCase):
    """Tests for Anthropic OAuth helpers and strategy."""

    def test_oauth_constants_defined(self):
        """OAuth constants should be properly defined."""
        from litellm.llms.anthropic.claude_oauth import (
            ANTHROPIC_AUTH_URL,
            ANTHROPIC_TOKEN_URL,
            ANTHROPIC_CLIENT_ID,
            ANTHROPIC_REDIRECT_URI,
            ANTHROPIC_SCOPES,
        )
        
        self.assertIn("claude.ai", ANTHROPIC_AUTH_URL)
        self.assertIn("console.anthropic.com", ANTHROPIC_TOKEN_URL)
        self.assertIsInstance(ANTHROPIC_CLIENT_ID, str)
        self.assertIn("localhost", ANTHROPIC_REDIRECT_URI)
        self.assertIn("user:profile", ANTHROPIC_SCOPES)

    def test_generate_auth_url(self):
        """Auth URL generation should produce valid URL."""
        from litellm.llms.anthropic.claude_oauth import generate_auth_url
        from litellm.auth.pkce import generate_pkce_pair, generate_state
        
        code_verifier, code_challenge = generate_pkce_pair()
        state = generate_state()
        
        url = generate_auth_url(state, code_challenge)
        
        self.assertIn("claude.ai/oauth/authorize", url)
        self.assertIn("client_id=", url)
        self.assertIn("code_challenge=", url)
        self.assertIn("state=", url)
        self.assertIn("code_challenge_method=S256", url)

    def test_token_data_dataclass(self):
        """Token data should work as a dataclass."""
        from litellm.llms.anthropic.claude_oauth import AnthropicTokenData
        
        token = AnthropicTokenData(
            access_token="test-token",
            refresh_token="test-refresh",
            email="test@example.com",
        )
        self.assertEqual(token.access_token, "test-token")
        self.assertEqual(token.email, "test@example.com")
        self.assertEqual(token.token_type, "Bearer")

    def test_auth_error(self):
        """Auth error should format message correctly."""
        from litellm.llms.anthropic.claude_oauth import AnthropicAuthError
        
        err = AnthropicAuthError("token_exchange_failed", "Invalid code")
        self.assertEqual(err.error_type, "token_exchange_failed")
        self.assertIn("token_exchange_failed", str(err))
        self.assertIn("Invalid code", str(err))

    def test_strategy_supports_claude(self):
        """Adapter should support claude models."""
        from litellm.auth.adapters.anthropic import AnthropicSubscriptionAdapter
        
        adapter = AnthropicSubscriptionAdapter()
        self.assertTrue(adapter.supports("claude-3"))
        self.assertTrue(adapter.supports("anthropic/claude-3-opus"))
        self.assertTrue(adapter.supports("Claude-Sonnet"))


if __name__ == "__main__":
    unittest.main()
