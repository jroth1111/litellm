"""
Cookie-based OAuth fallback for Anthropic Claude.

This module provides an alternative authentication mechanism for users who
already have an active browser session with Claude. Instead of going through
the full browser OAuth flow, users can provide their session key cookie to
obtain OAuth tokens.

This is an OPTIONAL alternative to the standard browser PKCE flow.

Usage:
    from litellm.llms.anthropic.cookie_oauth import (
        get_organizations,
        authorize_with_cookie,
    )
    
    # Get available organizations using session cookie
    session_key = "sk-ant-sid01-..."  # from browser cookies
    orgs = get_organizations(session_key)
    
    # Authorize with a specific organization
    tokens = authorize_with_cookie(
        session_key=session_key,
        organization_uuid=orgs[0]["uuid"],
    )

Reference: claude-relay-service oauthHelper.js
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx

from litellm.auth.pkce import generate_pkce_pair, generate_state
from litellm.auth.provider_http import http_get_with_retry, http_post_with_retry

# Cookie OAuth endpoints
CLAUDE_API_BASE = os.getenv("CLAUDE_API_BASE", "https://claude.ai")
ORGANIZATIONS_URL = f"{CLAUDE_API_BASE}/api/organizations"
AUTHORIZE_URL_TEMPLATE = f"{CLAUDE_API_BASE}/v1/oauth/{{organization_uuid}}/authorize"

# Default OAuth settings for cookie flow
COOKIE_OAUTH_CLIENT_ID = os.getenv(
    "ANTHROPIC_COOKIE_CLIENT_ID",
    "9d1c250a-e61b-44d9-88ed-5944d1962f5e"  # Same as standard Anthropic client
)
COOKIE_OAUTH_SCOPES = os.getenv(
    "ANTHROPIC_COOKIE_SCOPES",
    "org:create_api_key user:profile user:inference"
)
COOKIE_REDIRECT_URI = os.getenv(
    "ANTHROPIC_COOKIE_REDIRECT_URI",
    "http://localhost:54545/callback"
)


@dataclass
class Organization:
    """Claude organization info."""
    uuid: str
    name: str
    is_personal: bool = False
    role: Optional[str] = None


@dataclass
class CookieAuthResult:
    """Result from cookie-based authorization."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_at: Optional[str] = None
    organization_uuid: Optional[str] = None
    organization_name: Optional[str] = None
    email: Optional[str] = None


class CookieAuthError(Exception):
    """Error during cookie-based authentication."""
    
    def __init__(self, error_type: str, description: str = ""):
        self.error_type = error_type
        self.description = description
        super().__init__(f"{error_type}: {description}" if description else error_type)


def _session_headers(session_key: str) -> Dict[str, str]:
    """Build headers with session cookie."""
    return {
        "Cookie": f"sessionKey={session_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "LiteLLM-OAuth/1.0",
    }


def get_organizations(
    session_key: str,
    *,
    timeout: float = 30.0,
    proxy: Optional[str] = None,
) -> List[Organization]:
    """
    Fetch available organizations using session cookie.
    
    Args:
        session_key: The sessionKey cookie value from browser.
        timeout: HTTP request timeout.
        proxy: Optional proxy URL.
    
    Returns:
        List of Organization objects the user has access to.
    
    Raises:
        CookieAuthError: If the request fails or session is invalid.
    """
    try:
        resp = http_get_with_retry(
            ORGANIZATIONS_URL,
            headers=_session_headers(session_key),
            timeout=timeout,
            proxy=proxy,
        )
    except Exception as e:
        raise CookieAuthError("request_failed", str(e)) from e
    
    if resp.status_code == 401:
        raise CookieAuthError(
            "session_invalid",
            "Session key is invalid or expired. Please get a fresh session key from your browser."
        )
    
    if resp.status_code != 200:
        raise CookieAuthError(
            "request_failed",
            f"HTTP {resp.status_code}: {resp.text[:200]}"
        )
    
    try:
        data = resp.json()
    except Exception:
        raise CookieAuthError("invalid_response", "Failed to parse organizations response")
    
    orgs = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and "uuid" in item:
            orgs.append(Organization(
                uuid=item["uuid"],
                name=item.get("name", ""),
                is_personal=item.get("is_personal", False),
                role=item.get("role"),
            ))
    
    if not orgs:
        raise CookieAuthError(
            "no_organizations",
            "No organizations found. Ensure you have access to at least one Claude organization."
        )
    
    return orgs


def authorize_with_cookie(
    session_key: str,
    organization_uuid: str,
    *,
    scopes: str = COOKIE_OAUTH_SCOPES,
    client_id: str = COOKIE_OAUTH_CLIENT_ID,
    redirect_uri: str = COOKIE_REDIRECT_URI,
    timeout: float = 30.0,
    proxy: Optional[str] = None,
) -> CookieAuthResult:
    """
    Authorize using session cookie to get OAuth tokens.
    
    This bypasses the browser OAuth flow by using an existing session.
    
    Args:
        session_key: The sessionKey cookie value from browser.
        organization_uuid: UUID of the organization to authorize for.
        scopes: OAuth scopes to request.
        client_id: OAuth client ID.
        redirect_uri: OAuth redirect URI.
        timeout: HTTP request timeout.
        proxy: Optional proxy URL.
    
    Returns:
        CookieAuthResult with access token and optional refresh token.
    
    Raises:
        CookieAuthError: If authorization fails.
    """
    # Generate PKCE pair and state for the authorization
    code_verifier, code_challenge = generate_pkce_pair()
    state = generate_state()
    
    authorize_url = AUTHORIZE_URL_TEMPLATE.format(organization_uuid=organization_uuid)
    
    # Build authorization request body
    body = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scopes,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    
    try:
        resp = http_post_with_retry(
            authorize_url,
            json=body,
            headers=_session_headers(session_key),
            timeout=timeout,
            proxy=proxy,
        )
    except Exception as e:
        raise CookieAuthError("request_failed", str(e)) from e
    
    if resp.status_code == 401:
        raise CookieAuthError(
            "session_invalid",
            "Session key is invalid or expired."
        )
    
    if resp.status_code == 403:
        raise CookieAuthError(
            "access_denied",
            "Access denied. You may not have permission for this organization."
        )
    
    if resp.status_code not in (200, 201):
        try:
            error_data = resp.json()
            raise CookieAuthError(
                error_data.get("error", "authorization_failed"),
                error_data.get("error_description", resp.text[:200]),
            )
        except ValueError:
            raise CookieAuthError("authorization_failed", f"HTTP {resp.status_code}")
    
    try:
        result = resp.json()
    except Exception:
        raise CookieAuthError("invalid_response", "Failed to parse authorization response")
    
    if not result.get("access_token"):
        # May need to exchange code if we got one
        if result.get("code"):
            return _exchange_cookie_code(
                code=result["code"],
                code_verifier=code_verifier,
                state=state,
                redirect_uri=redirect_uri,
                client_id=client_id,
                timeout=timeout,
                proxy=proxy,
            )
        raise CookieAuthError("authorization_failed", "No access_token in response")
    
    # Calculate expiry
    expires_at = None
    if "expires_in" in result:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=int(result["expires_in"]))
        ).isoformat()
    
    return CookieAuthResult(
        access_token=result["access_token"],
        refresh_token=result.get("refresh_token"),
        token_type=result.get("token_type", "Bearer"),
        expires_at=expires_at,
        organization_uuid=organization_uuid,
        organization_name=result.get("organization", {}).get("name"),
        email=result.get("account", {}).get("email_address"),
    )


def _exchange_cookie_code(
    code: str,
    code_verifier: str,
    state: str,
    redirect_uri: str,
    client_id: str,
    timeout: float = 30.0,
    proxy: Optional[str] = None,
) -> CookieAuthResult:
    """Exchange authorization code for tokens (fallback if direct token not returned)."""
    from litellm.llms.anthropic.claude_oauth import (
        ANTHROPIC_TOKEN_URL,
        AnthropicAuthError,
    )
    
    body = {
        "code": code,
        "grant_type": "authorization_code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
        "state": state,
    }
    
    try:
        resp = http_post_with_retry(
            ANTHROPIC_TOKEN_URL,
            json=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=timeout,
            proxy=proxy,
        )
    except Exception as e:
        raise CookieAuthError("token_exchange_failed", str(e)) from e
    
    if resp.status_code != 200:
        try:
            error_data = resp.json()
            raise CookieAuthError(
                error_data.get("error", "token_exchange_failed"),
                error_data.get("error_description", resp.text[:200]),
            )
        except ValueError:
            raise CookieAuthError("token_exchange_failed", f"HTTP {resp.status_code}")
    
    result = resp.json()
    
    if not result.get("access_token"):
        raise CookieAuthError("token_exchange_failed", "No access_token in response")
    
    # Calculate expiry
    expires_at = None
    if "expires_in" in result:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=int(result["expires_in"]))
        ).isoformat()
    
    org = result.get("organization", {})
    account = result.get("account", {})
    
    return CookieAuthResult(
        access_token=result["access_token"],
        refresh_token=result.get("refresh_token"),
        token_type=result.get("token_type", "Bearer"),
        expires_at=expires_at,
        organization_uuid=org.get("uuid"),
        organization_name=org.get("name"),
        email=account.get("email_address"),
    )
