"""
Helper for Anthropic/Claude OAuth flows including browser-based authorization with PKCE.

This module provides utilities for initiating and completing the Anthropic OAuth
flow for Claude API access, ported from CLIProxyAPIPlus claude/anthropic_auth.go.

Usage:
    from litellm.llms.anthropic.claude_oauth import (
        generate_auth_url,
        exchange_code_for_tokens,
        refresh_tokens,
    )
    
    # Generate authorization URL with PKCE
    from litellm.auth.pkce import generate_pkce_pair, generate_state
    code_verifier, code_challenge = generate_pkce_pair()
    state = generate_state()
    auth_url = generate_auth_url(state, code_challenge)
    print(f"Open this URL in your browser: {auth_url}")
    
    # After user authorizes, exchange code for tokens
    # tokens = exchange_code_for_tokens(code, code_verifier)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx

from litellm.auth.provider_http import http_post_with_retry

from litellm.auth.provider_interfaces import (
    BearerRequestSigner,
    ProviderOAuth,
    ProviderTokenResult,
)

# OAuth constants from CLIProxyAPIPlus
ANTHROPIC_AUTH_URL = os.getenv("ANTHROPIC_AUTH_URL", "https://claude.ai/oauth/authorize")
ANTHROPIC_TOKEN_URL = os.getenv(
    "ANTHROPIC_TOKEN_URL", "https://console.anthropic.com/v1/oauth/token"
)
ANTHROPIC_CLIENT_ID = os.getenv("ANTHROPIC_CLIENT_ID", "9d1c250a-e61b-44d9-88ed-5944d1962f5e")
ANTHROPIC_REDIRECT_URI = os.getenv("ANTHROPIC_REDIRECT_URI", "http://localhost:54545/callback")
ANTHROPIC_SCOPES = os.getenv("ANTHROPIC_SCOPES", "org:create_api_key user:profile user:inference")


@dataclass
class AnthropicTokenData:
    """Token data from a successful Anthropic OAuth flow."""
    
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_at: Optional[str] = None  # ISO format
    email: Optional[str] = None
    organization_uuid: Optional[str] = None
    organization_name: Optional[str] = None


class AnthropicAuthError(Exception):
    """Error during Anthropic authentication."""
    
    def __init__(self, error_type: str, description: str = ""):
        self.error_type = error_type
        self.description = description
        super().__init__(f"{error_type}: {description}" if description else error_type)


def generate_auth_url(
    state: str,
    code_challenge: str,
    redirect_uri: str = ANTHROPIC_REDIRECT_URI,
    scopes: str = ANTHROPIC_SCOPES,
) -> str:
    """
    Generate the Anthropic OAuth authorization URL with PKCE.
    
    Args:
        state: CSRF protection state parameter.
        code_challenge: The PKCE code challenge (S256).
        redirect_uri: The callback URL for authorization.
        scopes: OAuth scopes to request.
    
    Returns:
        The authorization URL to redirect the user to.
    """
    params = {
        "code": "true",
        "client_id": ANTHROPIC_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scopes,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return f"{ANTHROPIC_AUTH_URL}?{urlencode(params)}"


def exchange_code_for_tokens(
    code: str,
    code_verifier: str,
    state: str,
    expected_state: Optional[str] = None,
    redirect_uri: str = ANTHROPIC_REDIRECT_URI,
    timeout: float = 30.0,
) -> AnthropicTokenData:
    """
    Exchange an authorization code for access and refresh tokens.
    
    Args:
        code: The authorization code from the OAuth callback.
        code_verifier: The PKCE code verifier used when generating the challenge.
        state: The state parameter (optional).
        expected_state: If provided, validate that `state` matches this value.
        redirect_uri: The redirect URI used in the authorization request.
        timeout: HTTP request timeout.
    
    Returns:
        AnthropicTokenData with access token and refresh token.
    
    Raises:
        AnthropicAuthError: If the token exchange fails.
    """
    # Handle code that may contain fragment with state
    parsed_code = code.split("#")[0] if "#" in code else code
    
    body = {
        "code": parsed_code,
        "grant_type": "authorization_code",
        "client_id": ANTHROPIC_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }
    if not state:
        raise AnthropicAuthError("missing_state", "State parameter is required")
    body["state"] = state
    if expected_state is not None and expected_state != state:
        raise AnthropicAuthError("state_mismatch", "State parameter mismatch")
    
    resp = http_post_with_retry(
        ANTHROPIC_TOKEN_URL,
        json=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        timeout=timeout,
    )
    
    if resp.status_code != 200:
        try:
            error_data = resp.json()
            raise AnthropicAuthError(
                error_data.get("error", "token_exchange_failed"),
                error_data.get("error_description", resp.text),
            )
        except ValueError:
            raise AnthropicAuthError("token_exchange_failed", f"HTTP {resp.status_code}")
    
    result = resp.json()
    
    if not result.get("access_token"):
        raise AnthropicAuthError("token_exchange_failed", "No access_token in response")
    
    # Calculate expiry
    expires_at = None
    if "expires_in" in result:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=int(result["expires_in"]))
        ).isoformat()
    
    # Extract organization and account info
    org = result.get("organization", {})
    account = result.get("account", {})
    
    return AnthropicTokenData(
        access_token=result["access_token"],
        refresh_token=result.get("refresh_token"),
        token_type=result.get("token_type", "Bearer"),
        expires_at=expires_at,
        email=account.get("email_address"),
        organization_uuid=org.get("uuid"),
        organization_name=org.get("name"),
    )


def refresh_tokens(
    refresh_token: str,
    timeout: float = 30.0,
    *,
    token_url: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> AnthropicTokenData:
    """
    Refresh Anthropic access token using a refresh token.
    
    Args:
        refresh_token: The refresh token from a previous authorization.
        timeout: HTTP request timeout.
    
    Returns:
        AnthropicTokenData with new access token.
    
    Raises:
        AnthropicAuthError: If token refresh fails.
    """
    body = {
        "client_id": client_id or ANTHROPIC_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    if client_secret:
        body["client_secret"] = client_secret
    
    resp = http_post_with_retry(
        token_url or ANTHROPIC_TOKEN_URL,
        json=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        timeout=timeout,
    )
    
    if resp.status_code != 200:
        try:
            error_data = resp.json()
            raise AnthropicAuthError(
                error_data.get("error", "refresh_failed"),
                error_data.get("error_description", resp.text),
            )
        except ValueError:
            raise AnthropicAuthError("refresh_failed", f"HTTP {resp.status_code}")
    
    result = resp.json()
    
    if not result.get("access_token"):
        raise AnthropicAuthError("refresh_failed", "No access_token in response")
    
    # Calculate expiry
    expires_at = None
    if "expires_in" in result:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=int(result["expires_in"]))
        ).isoformat()
    
    # Extract account info
    account = result.get("account", {})
    
    return AnthropicTokenData(
        access_token=result["access_token"],
        refresh_token=result.get("refresh_token", refresh_token),
        token_type=result.get("token_type", "Bearer"),
        expires_at=expires_at,
        email=account.get("email_address"),
    )


class AnthropicOAuthProvider(ProviderOAuth):
    """
    Thin ProviderOAuth adapter for Anthropic/Claude.

    Keeps legacy helpers intact while enabling a common provider interface.
    """

    provider = "anthropic"

    def __init__(self) -> None:
        self.signer = BearerRequestSigner()

    def authorize_url(
        self,
        *,
        state: str,
        code_challenge: Optional[str] = None,
        redirect_uri: Optional[str] = None,
    ) -> str:
        # code_challenge is required for Anthropic (PKCE); enforce presence.
        if not code_challenge:
            raise ValueError("code_challenge is required for Anthropic OAuth")
        return generate_auth_url(
            state=state,
            code_challenge=code_challenge,
            redirect_uri=redirect_uri or ANTHROPIC_REDIRECT_URI,
            scopes=ANTHROPIC_SCOPES,
        )

    def exchange_code(
        self,
        *,
        code: str,
        code_verifier: Optional[str] = None,
        state: Optional[str] = None,
        redirect_uri: Optional[str] = None,
    ) -> ProviderTokenResult:
        if not code_verifier:
            raise ValueError("code_verifier is required for Anthropic OAuth exchange")
        token = exchange_code_for_tokens(
            code=code,
            code_verifier=code_verifier,
            state=state,
            redirect_uri=redirect_uri or ANTHROPIC_REDIRECT_URI,
        )
        return ProviderTokenResult(
            access_token=token.access_token,
            refresh_token=token.refresh_token,
            expires_at=token.expires_at,
            token_type=token.token_type,
            metadata={
                "email": token.email,
                "organization_uuid": token.organization_uuid,
                "organization_name": token.organization_name,
            },
        )

    def refresh(self, *, refresh_token: str) -> ProviderTokenResult:
        token = refresh_tokens(refresh_token=refresh_token)
        return ProviderTokenResult(
            access_token=token.access_token,
            refresh_token=token.refresh_token,
            expires_at=token.expires_at,
            token_type=token.token_type,
            metadata={"email": token.email} if token.email else {},
        )


# Singleton-style instance for consumers that want the interface
anthropic_oauth_provider = AnthropicOAuthProvider()
