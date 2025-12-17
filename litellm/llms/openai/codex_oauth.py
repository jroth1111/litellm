"""
Helper for OpenAI Codex OAuth flows including browser-based authorization with PKCE.

This module provides utilities for initiating and completing the OpenAI OAuth
flow for Codex API access, ported from CLIProxyAPIPlus codex/openai_auth.go.

Usage:
    from litellm.llms.openai.codex_oauth import (
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

import logging

import httpx

from litellm.auth.provider_http import http_post_with_retry

logger = logging.getLogger("litellm.auth.oauth.openai")

# OAuth constants from CLIProxyAPIPlus
OPENAI_AUTH_URL = os.getenv("OPENAI_AUTH_URL", "https://auth.openai.com/oauth/authorize")
OPENAI_TOKEN_URL = os.getenv("OPENAI_TOKEN_URL", "https://auth.openai.com/oauth/token")
OPENAI_CLIENT_ID = os.getenv("OPENAI_CLIENT_ID", "app_EMoamEEZ73f0CkXaXp7hrann")
OPENAI_REDIRECT_URI = os.getenv("OPENAI_REDIRECT_URI", "http://localhost:1455/auth/callback")
OPENAI_SCOPES = os.getenv("OPENAI_SCOPES", "openid email profile offline_access")
OPENAI_EXPECTED_ISSUER = os.getenv("OPENAI_EXPECTED_ISSUER", "https://auth.openai.com")


@dataclass
class OpenAITokenData:
    """Token data from a successful OpenAI OAuth flow."""
    
    access_token: str
    refresh_token: Optional[str] = None
    id_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_at: Optional[str] = None  # ISO format
    email: Optional[str] = None
    account_id: Optional[str] = None


class OpenAIAuthError(Exception):
    """Error during OpenAI authentication."""
    
    def __init__(self, error_type: str, description: str = ""):
        self.error_type = error_type
        self.description = description
        super().__init__(f"{error_type}: {description}" if description else error_type)


def generate_auth_url(
    state: str,
    code_challenge: str,
    redirect_uri: str = OPENAI_REDIRECT_URI,
    scopes: str = OPENAI_SCOPES,
) -> str:
    """
    Generate the OpenAI OAuth authorization URL with PKCE.
    
    Args:
        state: CSRF protection state parameter.
        code_challenge: The PKCE code challenge (S256).
        redirect_uri: The callback URL for authorization.
        scopes: OAuth scopes to request.
    
    Returns:
        The authorization URL to redirect the user to.
    """
    params = {
        "client_id": OPENAI_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scopes,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "prompt": "login",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
    }
    return f"{OPENAI_AUTH_URL}?{urlencode(params)}"


def exchange_code_for_tokens(
    code: str,
    code_verifier: str,
    state: str,
    expected_state: Optional[str] = None,
    redirect_uri: str = OPENAI_REDIRECT_URI,
    timeout: float = 30.0,
) -> OpenAITokenData:
    """
    Exchange an authorization code for access and refresh tokens.
    
    Args:
        code: The authorization code from the OAuth callback.
        code_verifier: The PKCE code verifier used when generating the challenge.
        redirect_uri: The redirect URI used in the authorization request.
        timeout: HTTP request timeout.
    
    Returns:
        OpenAITokenData with access token, refresh token, and ID token.
    
    Raises:
        OpenAIAuthError: If the token exchange fails.
    """
    data = {
        "grant_type": "authorization_code",
        "client_id": OPENAI_CLIENT_ID,
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }
    if not state:
        raise OpenAIAuthError("missing_state", "State parameter is required")
    data["state"] = state
    if expected_state is not None and expected_state != state:
        raise OpenAIAuthError("state_mismatch", "State parameter mismatch")
    
    resp = http_post_with_retry(
        OPENAI_TOKEN_URL,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        timeout=timeout,
    )
    
    if resp.status_code != 200:
        try:
            error_data = resp.json()
            raise OpenAIAuthError(
                error_data.get("error", "token_exchange_failed"),
                error_data.get("error_description", resp.text),
            )
        except ValueError:
            raise OpenAIAuthError("token_exchange_failed", f"HTTP {resp.status_code}")
    
    result = resp.json()
    
    if not result.get("access_token"):
        raise OpenAIAuthError("token_exchange_failed", "No access_token in response")
    
    # Calculate expiry
    expires_at = None
    if "expires_in" in result:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=int(result["expires_in"]))
        ).isoformat()
    
    # Extract account info from ID token if present
    email = None
    account_id = None
    id_token = result.get("id_token")
    if id_token:
        # Verify ID token claims before extracting data
        if not verify_id_token(
            id_token,
            expected_issuer=OPENAI_EXPECTED_ISSUER,
            expected_audience=OPENAI_CLIENT_ID,
        ):
            logger.warning(
                "openai_id_token_verification_failed: ID token failed claim verification "
                "(expired, wrong issuer, or wrong audience). Claims will not be extracted."
            )
        else:
            # Simple JWT payload extraction (base64 decode middle part)
            try:
                import base64
                import json as json_module
                parts = id_token.split(".")
                if len(parts) >= 2:
                    # Add padding if necessary
                    payload = parts[1]
                    padding = 4 - (len(payload) % 4)
                    if padding != 4:
                        payload += "=" * padding
                    claims = json_module.loads(base64.urlsafe_b64decode(payload))
                    email = claims.get("email")
                    # Try different account ID fields
                    account_id = claims.get("sub") or claims.get("account_id")
            except Exception:
                pass  # ID token parsing is best-effort
    
    return OpenAITokenData(
        access_token=result["access_token"],
        refresh_token=result.get("refresh_token"),
        id_token=id_token,
        token_type=result.get("token_type", "Bearer"),
        expires_at=expires_at,
        email=email,
        account_id=account_id,
    )


def refresh_tokens(
    refresh_token: str,
    timeout: float = 30.0,
    *,
    token_url: Optional[str] = None,
    client_id: Optional[str] = None,
    scopes: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> OpenAITokenData:
    """
    Refresh OpenAI access token using a refresh token.
    
    Args:
        refresh_token: The refresh token from a previous authorization.
        timeout: HTTP request timeout.
    
    Returns:
        OpenAITokenData with new access token.
    
    Raises:
        OpenAIAuthError: If token refresh fails.
    """
    data = {
        "client_id": client_id or OPENAI_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "scope": scopes or OPENAI_SCOPES,
    }
    if client_secret:
        data["client_secret"] = client_secret
    
    resp = http_post_with_retry(
        token_url or OPENAI_TOKEN_URL,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        timeout=timeout,
    )
    
    if resp.status_code != 200:
        try:
            error_data = resp.json()
            raise OpenAIAuthError(
                error_data.get("error", "refresh_failed"),
                error_data.get("error_description", resp.text),
            )
        except ValueError:
            raise OpenAIAuthError("refresh_failed", f"HTTP {resp.status_code}")
    
    result = resp.json()
    
    if not result.get("access_token"):
        raise OpenAIAuthError("refresh_failed", "No access_token in response")
    
    # Calculate expiry
    expires_at = None
    if "expires_in" in result:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=int(result["expires_in"]))
        ).isoformat()
    
    # Extract account info from ID token if present
    email = None
    account_id = None
    id_token = result.get("id_token")
    if id_token:
        # Verify ID token claims before extracting data
        if not verify_id_token(
            id_token,
            expected_issuer=OPENAI_EXPECTED_ISSUER,
            expected_audience=OPENAI_CLIENT_ID,
        ):
            logger.warning(
                "openai_id_token_verification_failed: ID token failed claim verification "
                "(expired, wrong issuer, or wrong audience). Claims will not be extracted."
            )
        else:
            try:
                import base64
                import json as json_module
                parts = id_token.split(".")
                if len(parts) >= 2:
                    payload = parts[1]
                    padding = 4 - (len(payload) % 4)
                    if padding != 4:
                        payload += "=" * padding
                    claims = json_module.loads(base64.urlsafe_b64decode(payload))
                    email = claims.get("email")
                    account_id = claims.get("sub") or claims.get("account_id")
            except Exception:
                pass

    return OpenAITokenData(
        access_token=result["access_token"],
        refresh_token=result.get("refresh_token", refresh_token),
        id_token=id_token,
        token_type=result.get("token_type", "Bearer"),
        expires_at=expires_at,
        email=email,
        account_id=account_id,
    )


def verify_id_token(
    id_token: str,
    *,
    expected_audience: Optional[str] = None,
    expected_issuer: Optional[str] = None,
    leeway_seconds: int = 60,
) -> bool:
    """
    Lightweight, best-effort ID token verification (no signature verification).
    Checks exp/iss/aud claims if present.
    """
    try:
        import base64
        import json as json_module
        parts = id_token.split(".")
        if len(parts) < 2:
            return False
        payload = parts[1]
        padding = 4 - (len(payload) % 4)
        if padding != 4:
            payload += "=" * padding
        claims = json_module.loads(base64.urlsafe_b64decode(payload))

        from time import time as now

        exp = claims.get("exp")
        if exp and (now() - leeway_seconds) > float(exp):
            return False
        if expected_issuer and claims.get("iss") and claims.get("iss") != expected_issuer:
            return False
        if expected_audience:
            aud = claims.get("aud")
            if isinstance(aud, str) and aud != expected_audience:
                return False
            if isinstance(aud, list) and expected_audience not in aud:
                return False
        return True
    except Exception:
        return False
