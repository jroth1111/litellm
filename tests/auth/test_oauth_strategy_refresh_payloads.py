"""
Verify OAuth refresh payloads line up with CLIProxyAPIPlus defaults.

These tests assert that LiteLLM sends the same client identifiers and content
types the upstream examples expect for each provider.

Note: strategies delegate refresh behavior to provider helper modules in
`litellm/llms/*/*oauth*.py`. These tests validate the helper refresh payloads,
which keeps CLI login and proxy refresh behavior consistent.
"""

from __future__ import annotations

import json
import pathlib
import sys
import types
from urllib.parse import parse_qs

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


# Minimal package scaffolding so we can import helper modules without installing
# the full litellm dependency set.
if "litellm" not in sys.modules:
    pkg = types.ModuleType("litellm")
    pkg.__path__ = [str(ROOT / "litellm")]  # type: ignore[attr-defined]
    sys.modules["litellm"] = pkg
if "litellm.auth" not in sys.modules:
    auth_pkg = types.ModuleType("litellm.auth")
    auth_pkg.__path__ = [str(ROOT / "litellm" / "auth")]  # type: ignore[attr-defined]
    sys.modules["litellm.auth"] = auth_pkg


from litellm.llms.anthropic.claude_oauth import (  # noqa: E402
    ANTHROPIC_CLIENT_ID,
    ANTHROPIC_TOKEN_URL,
    refresh_tokens as anthropic_refresh_tokens,
)
from litellm.llms.antigravity.antigravity_oauth import (  # noqa: E402
    ANTIGRAVITY_CLIENT_ID,
    ANTIGRAVITY_TOKEN_URL,
    refresh_tokens as antigravity_refresh_tokens,
)
from litellm.llms.cursor.oauth_subscription import (  # noqa: E402
    CURSOR_REFRESH_URL,
    refresh_tokens as cursor_refresh_tokens,
)
from litellm.llms.gemini.oauth_subscription import (  # noqa: E402
    GEMINI_CLIENT_ID,
    GEMINI_TOKEN_URL,
    refresh_tokens as gemini_refresh_tokens,
)
from litellm.llms.openai.codex_oauth import (  # noqa: E402
    OPENAI_CLIENT_ID,
    OPENAI_SCOPES,
    OPENAI_TOKEN_URL,
    refresh_tokens as openai_refresh_tokens,
)
from litellm.llms.qwen.oauth_subscription import (  # noqa: E402
    QWEN_CLIENT_ID,
    QWEN_TOKEN_ENDPOINT,
    refresh_tokens as qwen_refresh_tokens,
)


@pytest.mark.respx()
def test_openai_refresh_uses_cli_proxy_defaults(respx_mock):
    respx_mock.post(OPENAI_TOKEN_URL).respond(
        json={"access_token": "tok", "refresh_token": "rt2", "expires_in": 1000}
    )

    refreshed = openai_refresh_tokens(refresh_token="rt")

    assert refreshed.access_token == "tok"
    req = respx_mock.calls.last.request
    body = parse_qs(req.content.decode())
    assert body["client_id"] == [OPENAI_CLIENT_ID]
    assert body["grant_type"] == ["refresh_token"]
    assert body["refresh_token"] == ["rt"]
    assert body["scope"] == [OPENAI_SCOPES]
    assert "application/x-www-form-urlencoded" in req.headers["content-type"]


@pytest.mark.respx()
def test_anthropic_refresh_sends_json_with_client_id(respx_mock):
    respx_mock.post(ANTHROPIC_TOKEN_URL).respond(
        json={"access_token": "tok", "refresh_token": "rt2", "expires_in": 3600}
    )

    refreshed = anthropic_refresh_tokens(refresh_token="rt")

    assert refreshed.access_token == "tok"
    req = respx_mock.calls.last.request
    payload = json.loads(req.content.decode())
    assert payload["client_id"] == ANTHROPIC_CLIENT_ID
    assert payload["grant_type"] == "refresh_token"
    assert payload["refresh_token"] == "rt"
    assert "application/json" in req.headers["content-type"]


@pytest.mark.respx()
def test_gemini_refresh_includes_client_credentials(respx_mock):
    respx_mock.post(GEMINI_TOKEN_URL).respond(
        json={"access_token": "tok", "refresh_token": "rt2", "expires_in": 1200}
    )

    refreshed = gemini_refresh_tokens(refresh_token="rt", client_secret="test-secret")

    assert refreshed.access_token == "tok"
    req = respx_mock.calls.last.request
    body = parse_qs(req.content.decode())
    assert body["client_id"] == [GEMINI_CLIENT_ID]
    assert body["client_secret"] == ["test-secret"]
    assert "application/x-www-form-urlencoded" in req.headers["content-type"]


@pytest.mark.respx()
def test_antigravity_refresh_includes_google_credentials(respx_mock):
    respx_mock.post(ANTIGRAVITY_TOKEN_URL).respond(
        json={"access_token": "tok", "refresh_token": "rt2", "expires_in": 900}
    )

    refreshed = antigravity_refresh_tokens(refresh_token="rt", client_secret="test-secret")

    assert refreshed.access_token == "tok"
    req = respx_mock.calls.last.request
    body = parse_qs(req.content.decode())
    assert body["client_id"] == [ANTIGRAVITY_CLIENT_ID]
    assert body["client_secret"] == ["test-secret"]
    assert "application/x-www-form-urlencoded" in req.headers["content-type"]


@pytest.mark.respx()
def test_qwen_refresh_keeps_default_client_id(respx_mock):
    respx_mock.post(QWEN_TOKEN_ENDPOINT).respond(
        json={"access_token": "tok", "refresh_token": "rt2", "expires_in": 400}
    )

    refreshed = qwen_refresh_tokens(refresh_token="rt")

    assert refreshed.access_token == "tok"
    req = respx_mock.calls.last.request
    body = parse_qs(req.content.decode())
    assert body["client_id"] == [QWEN_CLIENT_ID]
    assert body["grant_type"] == ["refresh_token"]
    assert "application/x-www-form-urlencoded" in req.headers["content-type"]


@pytest.mark.respx()
def test_cursor_refresh_sends_json_refresh_token(respx_mock):
    respx_mock.post(CURSOR_REFRESH_URL).respond(
        json={"accessToken": "tok", "refreshToken": "rt2", "expiresIn": 1000}
    )

    refreshed = cursor_refresh_tokens(refresh_token="rt")

    assert refreshed.access_token == "tok"
    req = respx_mock.calls.last.request
    payload = json.loads(req.content.decode())
    assert payload["refresh_token"] == "rt"
    assert "application/json" in req.headers["content-type"]

