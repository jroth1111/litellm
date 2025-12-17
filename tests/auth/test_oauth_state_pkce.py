"""
Contract tests for state validation and optional PKCE additions.

Uses a dummy aiohttp shim to allow importing litellm submodules without the full
dependency set in lightweight environments.
"""

import pytest


def test_anthropic_state_mismatch():
    from litellm.llms.anthropic.claude_oauth import AnthropicAuthError, exchange_code_for_tokens

    with pytest.raises(AnthropicAuthError) as exc:
        exchange_code_for_tokens(
            code="abc",
            code_verifier="verifier",
            state="foo",
            expected_state="bar",
        )
    assert exc.value.error_type == "state_mismatch"


def test_openai_state_mismatch():
    from litellm.llms.openai.codex_oauth import OpenAIAuthError, exchange_code_for_tokens

    with pytest.raises(OpenAIAuthError) as exc:
        exchange_code_for_tokens(
            code="abc",
            code_verifier="verifier",
            state="foo",
            expected_state="bar",
        )
    assert exc.value.error_type == "state_mismatch"


def test_gemini_pkce_and_state():
    from litellm.llms.gemini.oauth_subscription import (
        GeminiAuthError,
        exchange_code_for_tokens,
        generate_auth_url,
    )

    url = generate_auth_url(state="s", code_challenge="cc")
    assert "code_challenge=cc" in url
    assert "code_challenge_method=S256" in url

    with pytest.raises(GeminiAuthError) as exc:
        exchange_code_for_tokens(
            code="abc",
            state="foo",
            expected_state="bar",
        )
    assert exc.value.error_type == "state_mismatch"


def test_antigravity_pkce_and_state():
    from litellm.llms.antigravity.antigravity_oauth import (
        AntigravityAuthError,
        exchange_code_for_tokens,
        generate_auth_url,
    )

    url = generate_auth_url(state="s", code_challenge="cc")
    assert "code_challenge=cc" in url
    assert "code_challenge_method=S256" in url

    with pytest.raises(AntigravityAuthError) as exc:
        exchange_code_for_tokens(
            code="abc",
            state="foo",
            expected_state="bar",
        )
    assert exc.value.error_type == "state_mismatch"
