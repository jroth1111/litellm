"""
Helper for Antigravity OAuth flows using Google OAuth2.

This module provides utilities for initiating and completing the Antigravity OAuth
flow for Cloud Code Assist API access, ported from CLIProxyAPIPlus sdk/auth/antigravity.go.

Antigravity uses Google OAuth2 with specific scopes for cloud-platform and 
Google Cloud Code capabilities.

Usage:
    from litellm.llms.antigravity.antigravity_oauth import (
        generate_auth_url,
        exchange_code_for_tokens,
        refresh_tokens,
        fetch_user_info,
        fetch_project_id,
    )
    
    # Generate authorization URL
    from litellm.auth.pkce import generate_state
    state = generate_state()
    auth_url = generate_auth_url(state)
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

# OAuth constants from CLIProxyAPIPlus sdk/auth/antigravity.go with env overrides
ANTIGRAVITY_CLIENT_ID = os.getenv(
    "ANTIGRAVITY_CLIENT_ID",
    "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com",
)
ANTIGRAVITY_CLIENT_SECRET = os.getenv("ANTIGRAVITY_CLIENT_SECRET")
ANTIGRAVITY_CALLBACK_PORT = int(os.getenv("ANTIGRAVITY_CALLBACK_PORT", "51121"))
ANTIGRAVITY_REDIRECT_URI = os.getenv(
    "ANTIGRAVITY_REDIRECT_URI", f"http://localhost:{ANTIGRAVITY_CALLBACK_PORT}/oauth-callback"
)

# Uses Google OAuth2 endpoints
ANTIGRAVITY_AUTH_URL = os.getenv("ANTIGRAVITY_AUTH_URL", "https://accounts.google.com/o/oauth2/v2/auth")
ANTIGRAVITY_TOKEN_URL = os.getenv("ANTIGRAVITY_TOKEN_URL", "https://oauth2.googleapis.com/token")
ANTIGRAVITY_USERINFO_URL = os.getenv("ANTIGRAVITY_USERINFO_URL", "https://www.googleapis.com/oauth2/v1/userinfo")

# Scopes for Antigravity/Cloud Code
ANTIGRAVITY_SCOPES = os.getenv(
    "ANTIGRAVITY_SCOPES",
    "https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/cclog https://www.googleapis.com/auth/experimentsandconfigs",
).split()

# API constants for project discovery
ANTIGRAVITY_API_ENDPOINT = "https://cloudcode-pa.googleapis.com"
ANTIGRAVITY_API_VERSION = "v1internal"


@dataclass
class AntigravityTokenData:
    """Token data from a successful Antigravity OAuth flow."""
    
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_at: Optional[str] = None  # ISO format
    email: Optional[str] = None
    project_id: Optional[str] = None


class AntigravityAuthError(Exception):
    """Error during Antigravity authentication."""
    
    def __init__(self, error_type: str, description: str = ""):
        self.error_type = error_type
        self.description = description
        super().__init__(f"{error_type}: {description}" if description else error_type)


def generate_auth_url(
    state: str,
    redirect_uri: str = ANTIGRAVITY_REDIRECT_URI,
    scopes: Optional[List[str]] = None,
    code_challenge: Optional[str] = None,
) -> str:
    """
    Generate the Antigravity OAuth authorization URL (Google OAuth2).
    
    Args:
        state: CSRF protection state parameter.
        redirect_uri: The callback URL for authorization.
        scopes: OAuth scopes to request (defaults to ANTIGRAVITY_SCOPES).
    
    Returns:
        The authorization URL to redirect the user to.
    """
    if scopes is None:
        scopes = ANTIGRAVITY_SCOPES
    
    params = {
        "access_type": "offline",
        "client_id": ANTIGRAVITY_CLIENT_ID,
        "prompt": "consent",
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "state": state,
    }
    if code_challenge:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = "S256"
    return f"{ANTIGRAVITY_AUTH_URL}?{urlencode(params)}"


def exchange_code_for_tokens(
    code: str,
    state: str,
    redirect_uri: str = ANTIGRAVITY_REDIRECT_URI,
    code_verifier: Optional[str] = None,
    expected_state: Optional[str] = None,
    timeout: float = 30.0,
) -> AntigravityTokenData:
    """
    Exchange an authorization code for access and refresh tokens.
    
    Args:
        code: The authorization code from the OAuth callback.
        redirect_uri: The redirect URI used in the authorization request.
        timeout: HTTP request timeout.
    
    Returns:
        AntigravityTokenData with access token and refresh token.
    
    Raises:
        AntigravityAuthError: If the token exchange fails.
    """
    data = {
        "code": code,
        "client_id": ANTIGRAVITY_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    if ANTIGRAVITY_CLIENT_SECRET:
        data["client_secret"] = ANTIGRAVITY_CLIENT_SECRET
    if not state:
        raise AntigravityAuthError("missing_state", "State parameter is required")
    data["state"] = state
    if code_verifier:
        data["code_verifier"] = code_verifier
    if expected_state is not None and expected_state != state:
        raise AntigravityAuthError("state_mismatch", "State parameter mismatch")
    
    resp = http_post_with_retry(
        ANTIGRAVITY_TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout,
    )
    
    if resp.status_code != 200:
        try:
            error_data = resp.json()
            raise AntigravityAuthError(
                error_data.get("error", "token_exchange_failed"),
                error_data.get("error_description", resp.text),
            )
        except ValueError:
            raise AntigravityAuthError("token_exchange_failed", f"HTTP {resp.status_code}")
    
    result = resp.json()
    
    if not result.get("access_token"):
        raise AntigravityAuthError("token_exchange_failed", "No access_token in response")
    
    # Calculate expiry
    expires_at = None
    if "expires_in" in result:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=int(result["expires_in"]))
        ).isoformat()
    
    return AntigravityTokenData(
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
) -> AntigravityTokenData:
    """
    Refresh Antigravity access token using a refresh token.
    
    Args:
        refresh_token: The refresh token from a previous authorization.
        timeout: HTTP request timeout.
    
    Returns:
        AntigravityTokenData with new access token.
    
    Raises:
        AntigravityAuthError: If token refresh fails.
    """
    data = {
        "client_id": client_id or ANTIGRAVITY_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    effective_secret = client_secret or ANTIGRAVITY_CLIENT_SECRET
    if effective_secret:
        data["client_secret"] = effective_secret
    
    resp = http_post_with_retry(
        token_url or ANTIGRAVITY_TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout,
    )
    
    if resp.status_code != 200:
        try:
            error_data = resp.json()
            raise AntigravityAuthError(
                error_data.get("error", "refresh_failed"),
                error_data.get("error_description", resp.text),
            )
        except ValueError:
            raise AntigravityAuthError("refresh_failed", f"HTTP {resp.status_code}")
    
    result = resp.json()
    
    if not result.get("access_token"):
        raise AntigravityAuthError("refresh_failed", "No access_token in response")
    
    # Calculate expiry
    expires_at = None
    if "expires_in" in result:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=int(result["expires_in"]))
        ).isoformat()
    
    return AntigravityTokenData(
        access_token=result["access_token"],
        refresh_token=result.get("refresh_token", refresh_token),
        token_type=result.get("token_type", "Bearer"),
        expires_at=expires_at,
    )


def fetch_user_info(
    access_token: str,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    Fetch user info from Google userinfo endpoint.
    
    Args:
        access_token: Valid access token.
        timeout: HTTP request timeout.
    
    Returns:
        Dict with user info including 'email'.
    """
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(
            f"{ANTIGRAVITY_USERINFO_URL}?alt=json",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        
        if resp.status_code != 200:
            return {}
        
        return resp.json()


def fetch_project_id(
    access_token: str,
    timeout: float = 30.0,
) -> Optional[str]:
    """
    Fetch the GCP project ID via the loadCodeAssist endpoint.
    
    Args:
        access_token: Valid access token.
        timeout: HTTP request timeout.
    
    Returns:
        The cloudaicompanionProject ID, or None if not found.
    """
    endpoint_url = f"{ANTIGRAVITY_API_ENDPOINT}/{ANTIGRAVITY_API_VERSION}:loadCodeAssist"
    
    body = {
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }
    }
    
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            endpoint_url,
            json=body,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "User-Agent": "google-api-nodejs-client/9.15.1",
                "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
            },
        )
        
        if resp.status_code != 200:
            return None
        
        result = resp.json()
    
    # Extract project ID
    project_id = result.get("cloudaicompanionProject")
    if isinstance(project_id, str):
        return project_id.strip() or None
    if isinstance(project_id, dict):
        return project_id.get("id", "").strip() or None
    
    return None
