"""
Helper for Gemini OAuth flows including browser-based authorization.

This module provides utilities for initiating and completing the Google OAuth
flow for Gemini API access, ported from CLIProxyAPIPlus gemini/gemini_auth.go.

Usage:
    from litellm.llms.gemini.oauth_subscription import (
        generate_auth_url,
        exchange_code_for_tokens,
        refresh_tokens,
        fetch_user_info,
    )
    
    # Generate authorization URL
    auth_url = generate_auth_url(redirect_uri="http://localhost:8085/oauth2callback")
    print(f"Open this URL in your browser: {auth_url}")
    
    # After user authorizes, exchange code for tokens
    # tokens = exchange_code_for_tokens(code, redirect_uri)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx

from litellm.auth.provider_http import http_post_with_retry

from litellm.auth.core import AuthRecord, RequestContext

# Google OAuth constants from CLIProxyAPIPlus with env overrides
GEMINI_CLIENT_ID = os.getenv(
    "GEMINI_CLIENT_ID",
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com",
)
GEMINI_CLIENT_SECRET = os.getenv("GEMINI_CLIENT_SECRET")
GEMINI_AUTHORIZE_URL = os.getenv("GEMINI_AUTHORIZE_URL", "https://accounts.google.com/o/oauth2/v2/auth")
GEMINI_TOKEN_URL = os.getenv("GEMINI_TOKEN_URL", "https://oauth2.googleapis.com/token")
GEMINI_USERINFO_URL = os.getenv("GEMINI_USERINFO_URL", "https://www.googleapis.com/oauth2/v1/userinfo")
GEMINI_SCOPES: List[str] = (
    os.getenv(
        "GEMINI_SCOPES",
        "https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile",
    ).split()
)
GEMINI_DEFAULT_REDIRECT_URI = os.getenv("GEMINI_REDIRECT_URI", "http://localhost:8085/oauth2callback")


@dataclass
class GeminiTokenData:
    """Token data from a successful Gemini OAuth flow."""
    
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_at: Optional[str] = None  # ISO format
    email: Optional[str] = None
    project_id: Optional[str] = None


class GeminiAuthError(Exception):
    """Error during Gemini authentication."""
    
    def __init__(self, error_type: str, description: str = ""):
        self.error_type = error_type
        self.description = description
        super().__init__(f"{error_type}: {description}" if description else error_type)


def generate_auth_url(
    state: str,
    redirect_uri: str = GEMINI_DEFAULT_REDIRECT_URI,
    scopes: Optional[List[str]] = None,
    code_challenge: Optional[str] = None,
) -> str:
    """
    Generate the Google OAuth authorization URL.
    
    Args:
        redirect_uri: The callback URL for authorization.
        state: CSRF protection state parameter.
        scopes: List of OAuth scopes (defaults to GEMINI_SCOPES).
    
    Returns:
        The authorization URL to redirect the user to.
    """
    params = {
        "client_id": GEMINI_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes or GEMINI_SCOPES),
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    if code_challenge:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = "S256"
    return f"{GEMINI_AUTHORIZE_URL}?{urlencode(params)}"


def exchange_code_for_tokens(
    code: str,
    state: str,
    redirect_uri: str = GEMINI_DEFAULT_REDIRECT_URI,
    code_verifier: Optional[str] = None,
    expected_state: Optional[str] = None,
    timeout: float = 30.0,
) -> GeminiTokenData:
    """
    Exchange an authorization code for access and refresh tokens.
    
    Args:
        code: The authorization code from the OAuth callback.
        redirect_uri: The redirect URI used in the authorization request.
        timeout: HTTP request timeout.
    
    Returns:
        GeminiTokenData with access token and refresh token.
    
    Raises:
        GeminiAuthError: If the token exchange fails.
    """
    data = {
        "client_id": GEMINI_CLIENT_ID,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }
    if GEMINI_CLIENT_SECRET:
        data["client_secret"] = GEMINI_CLIENT_SECRET
    if not state:
        raise GeminiAuthError("missing_state", "State parameter is required")
    data["state"] = state
    if code_verifier:
        data["code_verifier"] = code_verifier
    if expected_state is not None and expected_state != state:
        raise GeminiAuthError("state_mismatch", "State parameter mismatch")
    
    resp = http_post_with_retry(GEMINI_TOKEN_URL, data=data, timeout=timeout)
    
    if resp.status_code != 200:
        try:
            error_data = resp.json()
            raise GeminiAuthError(
                error_data.get("error", "token_exchange_failed"),
                error_data.get("error_description", resp.text),
            )
        except ValueError:
            raise GeminiAuthError("token_exchange_failed", f"HTTP {resp.status_code}")
    
    result = resp.json()
    
    if not result.get("access_token"):
        raise GeminiAuthError("token_exchange_failed", "No access_token in response")
    
    # Calculate expiry
    expires_at = None
    if "expires_in" in result:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=int(result["expires_in"]))
        ).isoformat()
    
    return GeminiTokenData(
        access_token=result["access_token"],
        refresh_token=result.get("refresh_token"),
        token_type=result.get("token_type", "Bearer"),
        expires_at=expires_at,
    )


def refresh_tokens(
    refresh_token: str,
    timeout: float = 30.0,
    *,
    token_url: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> GeminiTokenData:
    """
    Refresh Gemini access token using a refresh token.
    
    Args:
        refresh_token: The refresh token from a previous authorization.
        timeout: HTTP request timeout.
    
    Returns:
        GeminiTokenData with new access token.
    
    Raises:
        GeminiAuthError: If token refresh fails.
    """
    data = {
        "client_id": client_id or GEMINI_CLIENT_ID,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    effective_secret = client_secret or GEMINI_CLIENT_SECRET
    if effective_secret:
        data["client_secret"] = effective_secret
    
    resp = http_post_with_retry(token_url or GEMINI_TOKEN_URL, data=data, timeout=timeout)
    
    if resp.status_code != 200:
        try:
            error_data = resp.json()
            raise GeminiAuthError(
                error_data.get("error", "refresh_failed"),
                error_data.get("error_description", resp.text),
            )
        except ValueError:
            raise GeminiAuthError("refresh_failed", f"HTTP {resp.status_code}")
    
    result = resp.json()
    
    if not result.get("access_token"):
        raise GeminiAuthError("refresh_failed", "No access_token in response")
    
    # Calculate expiry
    expires_at = None
    if "expires_in" in result:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=int(result["expires_in"]))
        ).isoformat()
    
    return GeminiTokenData(
        access_token=result["access_token"],
        refresh_token=result.get("refresh_token", refresh_token),
        token_type=result.get("token_type", "Bearer"),
        expires_at=expires_at,
    )


def fetch_user_info(access_token: str, timeout: float = 30.0) -> Dict[str, Any]:
    """
    Fetch Google user info for the authenticated user.
    
    Args:
        access_token: The OAuth access token.
        timeout: HTTP request timeout.
    
    Returns:
        Dictionary with user info (email, name, etc.).
    
    Raises:
        GeminiAuthError: If the user info request fails.
    """
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(
            f"{GEMINI_USERINFO_URL}?alt=json",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        
        if resp.status_code != 200:
            raise GeminiAuthError("user_info_failed", f"HTTP {resp.status_code}")
        
        return resp.json()


def _auth_headers(auth: AuthRecord) -> Dict[str, str]:
    """Generate authorization headers for Gemini API calls."""
    token = auth.metadata.get("access_token")
    if not token:
        raise ValueError("missing access_token for gemini subscription")
    return {"Authorization": f"Bearer {token}"}


def post(
    auth: AuthRecord,
    url: str,
    json_body: Dict[str, Any],
    timeout: float = 30.0,
    ctx: Optional[RequestContext] = None,
) -> Dict[str, Any]:
    """
    Execute a Gemini API call using the subscription access token.
    """
    headers = _auth_headers(auth)
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=json_body, headers=headers)
        resp.raise_for_status()
        return resp.json()
