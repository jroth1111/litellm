"""
Unit tests for OpenAI Codex OAuth flow.
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


class TestOpenAIOAuth(unittest.TestCase):
    """Tests for OpenAI OAuth helpers and strategy."""

    def test_oauth_constants_defined(self):
        """OAuth constants should be properly defined."""
        from litellm.llms.openai.codex_oauth import (
            OPENAI_AUTH_URL,
            OPENAI_TOKEN_URL,
            OPENAI_CLIENT_ID,
            OPENAI_REDIRECT_URI,
            OPENAI_SCOPES,
        )
        
        self.assertIn("auth.openai.com", OPENAI_AUTH_URL)
        self.assertIn("auth.openai.com", OPENAI_TOKEN_URL)
        self.assertIsInstance(OPENAI_CLIENT_ID, str)
        self.assertIn("localhost", OPENAI_REDIRECT_URI)
        self.assertIn("openid", OPENAI_SCOPES)

    def test_generate_auth_url(self):
        """Auth URL generation should produce valid URL."""
        from litellm.llms.openai.codex_oauth import generate_auth_url
        from litellm.auth.pkce import generate_pkce_pair, generate_state
        
        code_verifier, code_challenge = generate_pkce_pair()
        state = generate_state()
        
        url = generate_auth_url(state, code_challenge)
        
        self.assertIn("auth.openai.com/oauth/authorize", url)
        self.assertIn("client_id=", url)
        self.assertIn("code_challenge=", url)
        self.assertIn("state=", url)
        self.assertIn("code_challenge_method=S256", url)

    def test_token_data_dataclass(self):
        """Token data should work as a dataclass."""
        from litellm.llms.openai.codex_oauth import OpenAITokenData
        
        token = OpenAITokenData(
            access_token="test-token",
            refresh_token="test-refresh",
            id_token="test-id-token",
            email="test@example.com",
        )
        self.assertEqual(token.access_token, "test-token")
        self.assertEqual(token.id_token, "test-id-token")
        self.assertEqual(token.token_type, "Bearer")

    def test_auth_error(self):
        """Auth error should format message correctly."""
        from litellm.llms.openai.codex_oauth import OpenAIAuthError
        
        err = OpenAIAuthError("token_exchange_failed", "Invalid code")
        self.assertEqual(err.error_type, "token_exchange_failed")
        self.assertIn("token_exchange_failed", str(err))

    def test_strategy_supports_gpt(self):
        """Adapter should support GPT models."""
        from litellm.auth.adapters.openai_chatgpt import OpenAIChatGPTSubscriptionAdapter
        
        adapter = OpenAIChatGPTSubscriptionAdapter()
        self.assertTrue(adapter.supports("gpt-4"))
        self.assertTrue(adapter.supports("openai/gpt-4-turbo"))
        self.assertTrue(adapter.supports("chatgpt"))


if __name__ == "__main__":
    unittest.main()
