"""
Helper for Qwen OAuth subscription flows using device code + PKCE.

This module provides utilities for initiating and completing the Qwen OAuth
device flow, which uses PKCE for enhanced security.

Usage:
    from litellm.llms.qwen.oauth_subscription import (
        initiate_device_flow,
        poll_for_token,
        refresh_tokens,
    )
    
    # Start device flow
    device_flow = initiate_device_flow()
    print(f"Go to {device_flow.verification_uri} and enter code: {device_flow.user_code}")
    
    # Poll for token
    token_data = poll_for_token(device_flow.device_code, device_flow.code_verifier)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import httpx

from litellm.auth.pkce import generate_pkce_pair
from litellm.auth.provider_http import (
    build_oauth_httpx_client,
    http_post_with_retry,
    resolve_oauth_proxy_url,
)

# OAuth 2.0 endpoints and configuration from CLIProxyAPIPlus with env overrides
QWEN_DEVICE_CODE_ENDPOINT = os.getenv(
    "QWEN_DEVICE_CODE_ENDPOINT", "https://chat.qwen.ai/api/v1/oauth2/device/code"
)
QWEN_TOKEN_ENDPOINT = os.getenv("QWEN_TOKEN_ENDPOINT", "https://chat.qwen.ai/api/v1/oauth2/token")
QWEN_CLIENT_ID = os.getenv("QWEN_CLIENT_ID", "f0304373b74a44d2b584a3fb70ca9e56")
QWEN_SCOPES = os.getenv("QWEN_SCOPES", "openid profile email model.completion")
QWEN_GRANT_TYPE_DEVICE = "urn:ietf:params:oauth:grant-type:device_code"

logger = logging.getLogger("litellm.auth.oauth.qwen")
if os.getenv("LITELLM_OAUTH_DEBUG"):
    logger.setLevel(logging.DEBUG)


def _build_client(timeout: float = 30.0, proxy: Optional[str] = None) -> httpx.Client:
    """
    Backwards-compatible shim. Prefer `litellm.auth.provider_http.build_oauth_httpx_client`.
    """
    proxy_url = resolve_oauth_proxy_url(proxy)
    if proxy_url:
        logger.debug("qwen_using_proxy", extra={"proxy": proxy_url.split("@")[-1]})
    return build_oauth_httpx_client(timeout=timeout, proxy=proxy_url)


@dataclass
class QwenDeviceFlow:
    """Response from the Qwen device authorization endpoint."""

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int
    code_verifier: str  # PKCE code verifier to use when polling


@dataclass
class QwenTokenData:
    """Token data from a successful Qwen OAuth flow."""

    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    resource_url: Optional[str] = None
    expires_at: Optional[str] = None  # ISO format


class QwenAuthError(Exception):
    """Error during Qwen authentication."""

    def __init__(self, error_type: str, description: str = ""):
        self.error_type = error_type
        self.description = description
        super().__init__(f"{error_type}: {description}" if description else error_type)


def initiate_device_flow(timeout: float = 30.0) -> QwenDeviceFlow:
    """
    Initiate the Qwen OAuth 2.0 device authorization flow with PKCE.

    Returns:
        QwenDeviceFlow with device code, user code, verification URI, and PKCE verifier.

    Raises:
        QwenAuthError: If the device code request fails.
    """
    # Generate PKCE pair
    code_verifier, code_challenge = generate_pkce_pair()

    data = {
        "client_id": QWEN_CLIENT_ID,
        "scope": QWEN_SCOPES,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }

    resp = http_post_with_retry(
        QWEN_DEVICE_CODE_ENDPOINT,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise QwenAuthError(
            "device_code_failed",
            f"HTTP {resp.status_code}: {resp.text}",
        )
    result = resp.json()

    if not result.get("device_code"):
        raise QwenAuthError("device_code_failed", "No device_code in response")

    return QwenDeviceFlow(
        device_code=result["device_code"],
        user_code=result["user_code"],
        verification_uri=result.get("verification_uri", ""),
        verification_uri_complete=result.get("verification_uri_complete", ""),
        expires_in=result.get("expires_in", 600),
        interval=result.get("interval", 5),
        code_verifier=code_verifier,
    )


def poll_for_token(
    device_code: str,
    code_verifier: str,
    max_wait_seconds: Optional[int] = None,
    max_attempts: int = 60,
    initial_interval: float = 5.0,
    timeout: float = 30.0,
) -> QwenTokenData:
    """
    Poll the Qwen token endpoint for a token after user authorization.

    Args:
        device_code: The device code from initiate_device_flow.
        code_verifier: The PKCE code verifier from initiate_device_flow.
        max_attempts: Maximum number of polling attempts (default 60 = 5 minutes).
        initial_interval: Starting poll interval in seconds.
        timeout: HTTP request timeout.

    Returns:
        QwenTokenData with access token and optional refresh token.

    Raises:
        QwenAuthError: If polling times out, token expires, or access is denied.
    """
    poll_interval = initial_interval
    deadline: Optional[float] = None
    if max_wait_seconds is not None:
        try:
            wait = int(max_wait_seconds)
        except Exception:
            wait = 0
        if wait <= 0:
            raise QwenAuthError("timeout", "Polling timed out waiting for user authorization")
        deadline = time.monotonic() + wait

    attempt = 0
    while True:
        attempt += 1
        if deadline is not None and time.monotonic() >= deadline:
            break
        if deadline is None and attempt > max_attempts:
            break

        data = {
            "grant_type": QWEN_GRANT_TYPE_DEVICE,
            "client_id": QWEN_CLIENT_ID,
            "device_code": device_code,
            "code_verifier": code_verifier,
        }

        resp = http_post_with_retry(
            QWEN_TOKEN_ENDPOINT,
            data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            timeout=timeout,
            max_attempts=1,  # polling loop handles retry/backoff semantics
        )

        if resp.status_code == 200:
            result = resp.json()
            if result.get("access_token"):
                expires_in = result.get("expires_in", 3600)
                expires_at = (
                    datetime.now(timezone.utc) + timedelta(seconds=expires_in)
                ).isoformat()

                return QwenTokenData(
                    access_token=result["access_token"],
                    refresh_token=result.get("refresh_token"),
                    token_type=result.get("token_type", "Bearer"),
                    resource_url=result.get("resource_url"),
                    expires_at=expires_at,
                )

        # Handle error responses
        try:
            error_data = resp.json()
            error_type = error_data.get("error", "")
            error_desc = error_data.get("error_description", "")

            if error_type == "authorization_pending":
                logger.debug("qwen_authorization_pending", extra={"interval": poll_interval, "attempt": attempt})
                time.sleep(poll_interval)
                continue
            elif error_type == "slow_down":
                poll_interval = min(poll_interval * 1.5, 10.0)
                logger.debug("qwen_slow_down", extra={"next_interval": poll_interval, "attempt": attempt})
                time.sleep(poll_interval)
                continue
            elif error_type == "expired_token":
                logger.warning("qwen_expired_token")
                raise QwenAuthError("expired_token", "Device code expired")
            elif error_type == "access_denied":
                logger.warning("qwen_access_denied")
                raise QwenAuthError("access_denied", "User denied authorization")
            else:
                raise QwenAuthError(error_type, error_desc)
        except (ValueError, KeyError):
            # Non-JSON response, retry
            time.sleep(poll_interval)
            continue

    raise QwenAuthError("timeout", "Polling timed out waiting for user authorization")


def refresh_tokens(
    refresh_token: str,
    timeout: float = 30.0,
    *,
    token_url: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> QwenTokenData:
    """
    Refresh Qwen access token using a refresh token.

    Args:
        refresh_token: The refresh token from a previous authorization.
        timeout: HTTP request timeout.

    Returns:
        QwenTokenData with new access token and optional new refresh token.

    Raises:
        QwenAuthError: If token refresh fails.
    """
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id or QWEN_CLIENT_ID,
    }
    if client_secret:
        data["client_secret"] = client_secret

    resp = http_post_with_retry(
        token_url or QWEN_TOKEN_ENDPOINT,
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
            raise QwenAuthError(
                error_data.get("error", "refresh_failed"),
                error_data.get("error_description", resp.text),
            )
        except ValueError:
            raise QwenAuthError("refresh_failed", f"HTTP {resp.status_code}")

    result = resp.json()

    if not result.get("access_token"):
        raise QwenAuthError("refresh_failed", "No access_token in response")

    expires_in = result.get("expires_in", 3600)
    expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    ).isoformat()

    return QwenTokenData(
        access_token=result["access_token"],
        refresh_token=result.get("refresh_token", refresh_token),
        token_type=result.get("token_type", "Bearer"),
        resource_url=result.get("resource_url"),
        expires_at=expires_at,
    )


def _auth_headers(access_token: str) -> Dict[str, str]:
    """Generate authorization headers for Qwen API calls."""
    return {"Authorization": f"Bearer {access_token}"}


def post(
    access_token: str,
    url: str,
    json_body: Dict[str, Any],
    timeout: float = 30.0,
    proxy: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a Qwen API call using the subscription access token.

    Args:
        access_token: The OAuth access token.
        url: The API endpoint URL.
        json_body: The JSON request body.
        timeout: HTTP request timeout.
        proxy: Proxy URL (optional, falls back to env vars).

    Returns:
        The JSON response as a dictionary.
    """
    headers = _auth_headers(access_token)
    with _build_client(timeout=timeout, proxy=proxy) as client:
        resp = client.post(url, json=json_body, headers=headers)
        resp.raise_for_status()
        return resp.json()
