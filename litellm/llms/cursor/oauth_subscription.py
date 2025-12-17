"""
Helper for Cursor AI subscription OAuth device-link polling flow.

Cursor uses a non-standard device-link flow:
1. Client generates UUID + verifier.
2. User opens login URL on authenticator.cursor.sh with those params.
3. Client polls api2.cursor.sh/auth/poll until access/refresh tokens are issued.

Usage:
    from litellm.llms.cursor.oauth_subscription import (
        start_cursor_login,
        poll_for_token,
        refresh_tokens,
    )

    session = start_cursor_login()
    print(f"Open this URL to login: {session.login_url}")
    tokens = poll_for_token(session)
"""

from __future__ import annotations

import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import httpx

from litellm.auth.provider_http import build_oauth_httpx_client, resolve_oauth_proxy_url

logger = logging.getLogger("litellm.auth.oauth.cursor")
if os.getenv("LITELLM_OAUTH_DEBUG"):
    logger.setLevel(logging.DEBUG)

# Base URLs with env overrides (avoid clashing with CURSOR_API_BASE used by BYOK)
CURSOR_AUTHENTICATOR_URL = os.getenv(
    "CURSOR_AUTHENTICATOR_URL", "https://authenticator.cursor.sh"
)
CURSOR_API_URL = os.getenv("CURSOR_API_URL", "https://api2.cursor.sh")

# Derived endpoints (override individually if needed)
CURSOR_LOGIN_URL = os.getenv(
    "CURSOR_LOGIN_URL", f"{CURSOR_AUTHENTICATOR_URL}/login"
)
CURSOR_POLL_URL = os.getenv("CURSOR_POLL_URL", f"{CURSOR_API_URL}/auth/poll")
CURSOR_REFRESH_URL = os.getenv(
    "CURSOR_REFRESH_URL", f"{CURSOR_API_URL}/auth/refresh"
)


def _build_client(timeout: float = 30.0, proxy: Optional[str] = None) -> httpx.Client:
    """
    Backwards-compatible shim. Prefer `litellm.auth.provider_http.build_oauth_httpx_client`.
    """
    proxy_url = resolve_oauth_proxy_url(proxy)
    if proxy_url:
        logger.debug("cursor_using_proxy", extra={"proxy": proxy_url.split("@")[-1]})
    return build_oauth_httpx_client(timeout=timeout, proxy=proxy_url)


@dataclass
class CursorLoginSession:
    """Ephemeral session details for a Cursor device-link login."""

    uuid: str
    verifier: str
    login_url: str


@dataclass
class CursorTokenData:
    """Token data from a successful Cursor OAuth flow."""

    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_at: Optional[str] = None  # ISO format
    raw: Optional[Dict[str, Any]] = None  # raw provider payload


class CursorAuthError(Exception):
    """Error during Cursor authentication."""

    def __init__(self, error_type: str, description: str = ""):
        self.error_type = error_type
        self.description = description
        super().__init__(
            f"{error_type}: {description}" if description else error_type
        )


def start_cursor_login(verifier_bytes: int = 32) -> CursorLoginSession:
    """
    Start a Cursor login session by generating UUID + verifier.

    Returns:
        CursorLoginSession containing login URL to present to the user.
    """
    session_uuid = str(uuid.uuid4())
    verifier = secrets.token_hex(verifier_bytes)
    login_url = f"{CURSOR_LOGIN_URL}?uuid={session_uuid}&verifier={verifier}"
    return CursorLoginSession(uuid=session_uuid, verifier=verifier, login_url=login_url)


def _parse_expiry(payload: Dict[str, Any]) -> Optional[str]:
    if not payload:
        return None
    if "expires_at" in payload:
        try:
            return str(payload["expires_at"])
        except Exception:
            pass
    expires_in = (
        payload.get("expiresIn")
        or payload.get("expires_in")
        or payload.get("expires")
    )
    if expires_in:
        try:
            exp_dt = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
            return exp_dt.isoformat()
        except Exception:
            return None
    return None


def poll_for_token(
    session: CursorLoginSession,
    *,
    max_wait_seconds: Optional[int] = None,
    max_attempts: int = 600,
    interval_seconds: float = 1.5,
    timeout: float = 30.0,
    proxy: Optional[str] = None,
) -> CursorTokenData:
    """
    Poll Cursor /auth/poll until user authorizes the session.

    Cursor returns 404 or 401 while pending; these are ignored.

    Args:
        session: The login session from start_cursor_login.
        max_wait_seconds: Maximum time to wait (overrides max_attempts).
        max_attempts: Maximum poll attempts.
        interval_seconds: Delay between polls.
        timeout: HTTP request timeout.
        proxy: Proxy URL (optional, falls back to env vars).
    """
    params = {"uuid": session.uuid, "verifier": session.verifier}
    effective_attempts = max_attempts
    if max_wait_seconds is not None:
        try:
            wait = int(max_wait_seconds)
        except Exception:
            wait = 0
        if wait > 0 and interval_seconds > 0:
            # Make max_wait_seconds authoritative when provided.
            effective_attempts = max(1, int((wait / interval_seconds) + 1))
    with _build_client(timeout=timeout, proxy=proxy) as client:
        for attempt in range(effective_attempts):
            resp = client.get(
                CURSOR_POLL_URL,
                params=params,
                headers={"Accept": "application/json"},
            )

            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except Exception as e:
                    raise CursorAuthError("poll_failed", f"invalid json: {e}") from e

                access_token = (
                    payload.get("accessToken")
                    or payload.get("access_token")
                    or payload.get("token")
                )
                refresh_token = (
                    payload.get("refreshToken") or payload.get("refresh_token")
                )
                if not access_token:
                    raise CursorAuthError(
                        "poll_failed", "missing access token in response"
                    )
                return CursorTokenData(
                    access_token=access_token,
                    refresh_token=refresh_token,
                    expires_at=_parse_expiry(payload),
                    raw=payload,
                )

            if resp.status_code in (401, 404):
                logger.debug(
                    "cursor_authorization_pending",
                    extra={"attempt": attempt, "interval": interval_seconds},
                )
                time.sleep(interval_seconds)
                continue

            if resp.status_code >= 500:
                logger.debug(
                    "cursor_poll_server_error",
                    extra={"attempt": attempt, "status": resp.status_code},
                )
                time.sleep(interval_seconds)
                continue

            raise CursorAuthError(
                "poll_failed", f"HTTP {resp.status_code}: {resp.text}"
            )

    raise CursorAuthError("timeout", "Polling timed out waiting for authorization")


def refresh_tokens(
    refresh_token: str,
    timeout: float = 30.0,
    *,
    token_url: Optional[str] = None,
    proxy: Optional[str] = None,
) -> CursorTokenData:
    """
    Refresh Cursor access token using refresh token.

    Args:
        refresh_token: The refresh token from a previous authorization.
        timeout: HTTP request timeout.
        token_url: Override token endpoint URL.
        proxy: Proxy URL (optional, falls back to env vars).
    """
    if not refresh_token:
        raise CursorAuthError("refresh_failed", "missing refresh_token")

    with _build_client(timeout=timeout, proxy=proxy) as client:
        resp = client.post(
            token_url or CURSOR_REFRESH_URL,
            json={"refresh_token": refresh_token},
            headers={"Accept": "application/json"},
        )

    if resp.status_code != 200:
        raise CursorAuthError(
            "refresh_failed", f"HTTP {resp.status_code}: {resp.text}"
        )

    payload = resp.json()
    access_token = (
        payload.get("accessToken")
        or payload.get("access_token")
        or payload.get("token")
    )
    new_refresh_token = payload.get("refreshToken") or payload.get("refresh_token")
    if not access_token:
        raise CursorAuthError("refresh_failed", "missing access token in response")

    return CursorTokenData(
        access_token=access_token,
        refresh_token=new_refresh_token or refresh_token,
        expires_at=_parse_expiry(payload),
        raw=payload,
    )
