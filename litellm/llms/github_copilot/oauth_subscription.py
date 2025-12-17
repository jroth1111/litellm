"""
Helper for GitHub Copilot OAuth flows including device code flow.

This module provides utilities for initiating and completing the GitHub Copilot
OAuth device flow, ported from CLIProxyAPIPlus copilot/oauth.go.

Usage:
    from litellm.llms.github_copilot.oauth_subscription import (
        request_device_code,
        poll_for_token,
        fetch_user_info,
    )
    
    # Start device flow
    device_code = request_device_code()
    print(f"Go to {device_code.verification_uri} and enter code: {device_code.user_code}")
    
    # Poll for token
    token_data = poll_for_token(device_code)
"""

from __future__ import annotations

import time
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import httpx

from litellm.auth.provider_http import http_get_with_retry, http_post_with_retry
from litellm.auth.core import AuthRecord, RequestContext


def _resolve_proxy_url(explicit_proxy: Optional[str] = None) -> Optional[str]:
    """
    Resolve proxy URL from explicit parameter or environment variables.

    Priority: explicit param > LITELLM_OAUTH_PROXY > HTTPS_PROXY > HTTP_PROXY.
    """
    if explicit_proxy:
        return explicit_proxy.strip() if explicit_proxy.strip() else None
    for var in ("LITELLM_OAUTH_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        val = os.getenv(var, "").strip()
        if val:
            return val
    return None


def _build_client(timeout: float = 30.0, proxy: Optional[str] = None) -> httpx.Client:
    """Build an httpx.Client with optional proxy support."""
    proxy_url = _resolve_proxy_url(proxy)
    if proxy_url:
        logger.debug("copilot_using_proxy", extra={"proxy": proxy_url.split("@")[-1]})
        return httpx.Client(timeout=timeout, proxy=proxy_url)
    return httpx.Client(timeout=timeout)

# OAuth constants from CLIProxyAPIPlus with env overrides
COPILOT_CLIENT_ID = os.getenv("COPILOT_CLIENT_ID", "Iv1.b507a08c87ecfe98")
COPILOT_DEVICE_CODE_URL = os.getenv("COPILOT_DEVICE_CODE_URL", "https://github.com/login/device/code")
COPILOT_TOKEN_URL = os.getenv("COPILOT_TOKEN_URL", "https://github.com/login/oauth/access_token")
COPILOT_USER_INFO_URL = os.getenv("COPILOT_USER_INFO_URL", "https://api.github.com/user")
COPILOT_DEFAULT_SCOPE = os.getenv("COPILOT_SCOPE", "user:email")
COPILOT_DEFAULT_POLL_INTERVAL = int(os.getenv("COPILOT_POLL_INTERVAL", "5"))
COPILOT_MAX_POLL_DURATION = int(os.getenv("COPILOT_MAX_POLL_DURATION", str(15 * 60)))

logger = logging.getLogger("litellm.auth.oauth.copilot")
if os.getenv("LITELLM_OAUTH_DEBUG"):
    logger.setLevel(logging.DEBUG)


@dataclass
class DeviceCodeResponse:
    """Response from GitHub's device code endpoint."""
    
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


@dataclass
class CopilotTokenData:
    """Token data from a successful Copilot OAuth flow."""
    
    access_token: str
    token_type: str = "Bearer"
    scope: str = ""


class CopilotAuthError(Exception):
    """Error during Copilot authentication."""
    
    def __init__(self, error_type: str, description: str = ""):
        self.error_type = error_type
        self.description = description
        super().__init__(f"{error_type}: {description}" if description else error_type)


def request_device_code(timeout: float = 30.0) -> DeviceCodeResponse:
    """
    Request a device code from GitHub for the Copilot OAuth flow.
    
    Returns:
        DeviceCodeResponse with device_code, user_code, and verification_uri.
    
    Raises:
        CopilotAuthError: If the device code request fails.
    """
    data = {
        "client_id": COPILOT_CLIENT_ID,
        "scope": COPILOT_DEFAULT_SCOPE,
    }
    
    resp = http_post_with_retry(
        COPILOT_DEVICE_CODE_URL,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        timeout=timeout,
    )
    if resp.status_code < 200 or resp.status_code >= 300:
        raise CopilotAuthError(
            "device_code_failed", f"HTTP {resp.status_code}: {resp.text}"
        )
    result = resp.json()
    
    return DeviceCodeResponse(
        device_code=result["device_code"],
        user_code=result["user_code"],
        verification_uri=result.get("verification_uri", "https://github.com/login/device"),
        expires_in=result.get("expires_in", 900),
        interval=result.get("interval", COPILOT_DEFAULT_POLL_INTERVAL),
    )


def poll_for_token(
    device_code: DeviceCodeResponse,
    timeout: float = 30.0,
    max_wait_seconds: Optional[int] = None,
) -> CopilotTokenData:
    """
    Poll the GitHub token endpoint until the user authorizes the device.
    
    Args:
        device_code: The device code response from request_device_code.
        timeout: HTTP request timeout.
    
    Returns:
        CopilotTokenData with access token.
    
    Raises:
        CopilotAuthError: If polling times out, code expires, or access is denied.
    """
    poll_interval = device_code.interval
    if poll_interval < COPILOT_DEFAULT_POLL_INTERVAL:
        poll_interval = COPILOT_DEFAULT_POLL_INTERVAL
    
    now = time.time()
    deadline = now + min(device_code.expires_in, COPILOT_MAX_POLL_DURATION)
    if max_wait_seconds is not None:
        try:
            max_wait = int(max_wait_seconds)
        except Exception:
            max_wait = 0
        if max_wait <= 0:
            raise CopilotAuthError("timeout", "Polling timed out waiting for authorization")
        deadline = min(deadline, now + max_wait)
    
    while time.time() < deadline:
        data = {
            "client_id": COPILOT_CLIENT_ID,
            "device_code": device_code.device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }

        resp = http_post_with_retry(
            COPILOT_TOKEN_URL,
            data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            timeout=timeout,
            max_attempts=1,  # polling loop handles retry/backoff semantics
        )

        result = resp.json()

        # Check for OAuth errors
        if "error" in result:
            error = result["error"]
            error_desc = result.get("error_description", "")

            if error == "authorization_pending":
                logger.debug("copilot_authorization_pending", extra={"attempt_interval": poll_interval})
                time.sleep(poll_interval)
                continue
            elif error == "slow_down":
                poll_interval += 5
                logger.debug("copilot_slow_down", extra={"next_interval": poll_interval})
                time.sleep(poll_interval)
                continue
            elif error == "expired_token":
                logger.warning("copilot_expired_token")
                raise CopilotAuthError("expired_token", "Device code expired")
            elif error == "access_denied":
                logger.warning("copilot_access_denied")
                raise CopilotAuthError("access_denied", "User denied authorization")
            else:
                raise CopilotAuthError(error, error_desc)

        # Success
        if "access_token" in result:
            return CopilotTokenData(
                access_token=result["access_token"],
                token_type=result.get("token_type", "Bearer"),
                scope=result.get("scope", ""),
            )

        time.sleep(poll_interval)
    
    raise CopilotAuthError("timeout", "Polling timed out waiting for authorization")


def fetch_user_info(access_token: str, timeout: float = 30.0) -> str:
    """
    Fetch the GitHub username for the authenticated user.
    
    Args:
        access_token: The OAuth access token.
        timeout: HTTP request timeout.
    
    Returns:
        The GitHub username (login).
    
    Raises:
        CopilotAuthError: If the user info request fails.
    """
    resp = http_get_with_retry(
        COPILOT_USER_INFO_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "User-Agent": "LiteLLM",
        },
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise CopilotAuthError("user_info_failed", f"HTTP {resp.status_code}")
    result = resp.json()
    
    username = result.get("login")
    if not username:
        raise CopilotAuthError("user_info_failed", "Empty username in response")
    
    return username


def _auth_headers(auth: AuthRecord) -> Dict[str, str]:
    """Generate authorization headers for Copilot API calls."""
    token = auth.metadata.get("access_token")
    if not token:
        raise ValueError("missing access_token for copilot subscription")
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def post(
    auth: AuthRecord,
    url: str,
    json_body: Dict[str, Any],
    timeout: float = 30.0,
    ctx: Optional[RequestContext] = None,
    proxy: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a Copilot API call using the subscription access token.

    Args:
        auth: AuthRecord with access token in metadata.
        url: API endpoint URL.
        json_body: JSON request body.
        timeout: HTTP request timeout.
        ctx: Request context (optional).
        proxy: Proxy URL (optional, falls back to env vars).
    """
    headers = _auth_headers(auth)
    with _build_client(timeout=timeout, proxy=proxy) as client:
        resp = client.post(url, json=json_body, headers=headers)
        resp.raise_for_status()
        return resp.json()
