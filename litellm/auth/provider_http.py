"""
Shared HTTP helpers for provider OAuth flows with minimal retry/backoff.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional, Union

import httpx

logger = logging.getLogger("litellm.auth.oauth")
if os.getenv("LITELLM_OAUTH_DEBUG"):
    logger.setLevel(logging.DEBUG)


def _resolve_proxy_url(explicit_proxy: Optional[str] = None) -> Optional[str]:
    """
    Resolve proxy URL from explicit parameter or environment variables.

    Priority:
    1. Explicit proxy parameter
    2. LITELLM_OAUTH_PROXY environment variable
    3. HTTPS_PROXY / https_proxy environment variable
    4. HTTP_PROXY / http_proxy environment variable

    Returns:
        Proxy URL string or None if no proxy configured.
    """
    if explicit_proxy:
        return explicit_proxy.strip() if explicit_proxy.strip() else None

    # Check LiteLLM-specific env var first
    oauth_proxy = os.getenv("LITELLM_OAUTH_PROXY", "").strip()
    if oauth_proxy:
        return oauth_proxy

    # Fall back to standard proxy env vars
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        val = os.getenv(var, "").strip()
        if val:
            return val

    return None


def resolve_oauth_proxy_url(explicit_proxy: Optional[str] = None) -> Optional[str]:
    """
    Public wrapper for resolving the proxy URL for OAuth HTTP calls.

    Keeps the internal resolution order stable while allowing other modules
    to reuse it without copy/paste.
    """
    return _resolve_proxy_url(explicit_proxy)


def _build_client(
    timeout: float = 30.0,
    proxy: Optional[str] = None,
) -> httpx.Client:
    """
    Build an httpx.Client with optional proxy support.

    Supports HTTP, HTTPS, and SOCKS5 proxies.

    Args:
        timeout: Request timeout in seconds.
        proxy: Proxy URL (http://host:port, https://host:port, socks5://host:port).
               If None, checks LITELLM_OAUTH_PROXY, then HTTPS_PROXY/HTTP_PROXY.

    Returns:
        Configured httpx.Client instance.
    """
    proxy_url = _resolve_proxy_url(proxy)

    if proxy_url:
        logger.debug("oauth_using_proxy", extra={"proxy": proxy_url.split("@")[-1]})  # mask credentials
        # httpx supports http, https, socks5 via the proxy parameter
        return httpx.Client(timeout=timeout, proxy=proxy_url)

    return httpx.Client(timeout=timeout)


def build_oauth_httpx_client(
    *,
    timeout: float = 30.0,
    proxy: Optional[str] = None,
) -> httpx.Client:
    """
    Public wrapper for building an httpx client for OAuth calls.
    """
    return _build_client(timeout=timeout, proxy=proxy)


def _retry_after_seconds(resp: httpx.Response) -> Optional[float]:
    """
    Parse Retry-After header from an HTTP response.

    Supports:
    - integer seconds
    - HTTP-date
    """
    try:
        raw = resp.headers.get("Retry-After")
    except Exception:
        raw = None
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.isdigit():
        try:
            seconds = float(s)
        except Exception:
            return None
        return seconds if seconds > 0 else None
    try:
        dt = parsedate_to_datetime(s)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        seconds = (dt - datetime.now(timezone.utc)).total_seconds()
        return seconds if seconds > 0 else None
    except Exception:
        return None


def http_post_with_retry(
    url: str,
    *,
    data: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    max_attempts: int = 3,
    backoff_seconds: float = 0.5,
    backoff_factor: float = 2.0,
    max_backoff_seconds: float = 10.0,
    retry_on_status: tuple = (429, 500, 502, 503, 504),
    proxy: Optional[str] = None,
) -> httpx.Response:
    """
    HTTP POST with retry and optional proxy support.

    Features:
    - Retries on transient status codes (429, 5xx)
    - Exponential backoff with configurable factor
    - Proxy support via parameter or environment variables
    - Connection errors trigger retry

    Args:
        url: Target URL.
        data: Form data to send.
        json: JSON body to send.
        headers: Request headers.
        timeout: Request timeout in seconds.
        max_attempts: Maximum retry attempts.
        backoff_seconds: Initial backoff delay.
        backoff_factor: Multiplier for exponential backoff.
        max_backoff_seconds: Maximum backoff delay cap.
        retry_on_status: Status codes that trigger retry.
        proxy: Proxy URL (optional, falls back to env vars).

    Returns:
        httpx.Response from the final attempt.

    Raises:
        httpx.ConnectError: If all connection attempts fail.
    """
    attempt = 0
    last_exception: Optional[Exception] = None
    current_backoff = backoff_seconds

    while attempt < max_attempts:
        attempt += 1
        start = time.monotonic()
        try:
            with _build_client(timeout=timeout, proxy=proxy) as client:
                resp = client.post(url, data=data, json=json, headers=headers)
            elapsed = time.monotonic() - start
            logger.debug(
                "oauth_http_post",
                extra={
                    "url": url,
                    "status_code": resp.status_code,
                    "attempt": attempt,
                    "elapsed_ms": round(elapsed * 1000, 1),
                },
            )
            if resp.status_code not in retry_on_status or attempt >= max_attempts:
                return resp
            retry_after = _retry_after_seconds(resp)
            if retry_after is not None:
                current_backoff = max(
                    current_backoff, min(float(retry_after), max_backoff_seconds)
                )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            elapsed = time.monotonic() - start
            logger.debug(
                "oauth_http_post_error",
                extra={
                    "url": url,
                    "attempt": attempt,
                    "error": str(e),
                    "elapsed_ms": round(elapsed * 1000, 1),
                },
            )
            last_exception = e
            if attempt >= max_attempts:
                raise

        # Exponential backoff
        time.sleep(min(current_backoff, max_backoff_seconds))
        current_backoff *= backoff_factor

    # Should not reach here, but satisfy type checker
    if last_exception:
        raise last_exception
    raise RuntimeError("http_post_with_retry exhausted attempts unexpectedly")


def http_get_with_retry(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
    max_attempts: int = 3,
    backoff_seconds: float = 0.5,
    backoff_factor: float = 2.0,
    max_backoff_seconds: float = 10.0,
    retry_on_status: tuple = (429, 500, 502, 503, 504),
    proxy: Optional[str] = None,
) -> httpx.Response:
    """
    HTTP GET with retry and optional proxy support.

    Same behavior as http_post_with_retry but for GET requests.
    Useful for fetching user info, device flow status, etc.

    Args:
        url: Target URL.
        headers: Request headers.
        params: Query parameters.
        timeout: Request timeout in seconds.
        max_attempts: Maximum retry attempts.
        backoff_seconds: Initial backoff delay.
        backoff_factor: Multiplier for exponential backoff.
        max_backoff_seconds: Maximum backoff delay cap.
        retry_on_status: Status codes that trigger retry.
        proxy: Proxy URL (optional, falls back to env vars).

    Returns:
        httpx.Response from the final attempt.

    Raises:
        httpx.ConnectError: If all connection attempts fail.
    """
    attempt = 0
    last_exception: Optional[Exception] = None
    current_backoff = backoff_seconds

    while attempt < max_attempts:
        attempt += 1
        start = time.monotonic()
        try:
            with _build_client(timeout=timeout, proxy=proxy) as client:
                resp = client.get(url, headers=headers, params=params)
            elapsed = time.monotonic() - start
            logger.debug(
                "oauth_http_get",
                extra={
                    "url": url,
                    "status_code": resp.status_code,
                    "attempt": attempt,
                    "elapsed_ms": round(elapsed * 1000, 1),
                },
            )
            if resp.status_code not in retry_on_status or attempt >= max_attempts:
                return resp
            retry_after = _retry_after_seconds(resp)
            if retry_after is not None:
                current_backoff = max(
                    current_backoff, min(float(retry_after), max_backoff_seconds)
                )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            elapsed = time.monotonic() - start
            logger.debug(
                "oauth_http_get_error",
                extra={
                    "url": url,
                    "attempt": attempt,
                    "error": str(e),
                    "elapsed_ms": round(elapsed * 1000, 1),
                },
            )
            last_exception = e
            if attempt >= max_attempts:
                raise

        # Exponential backoff
        time.sleep(min(current_backoff, max_backoff_seconds))
        current_backoff *= backoff_factor

    # Should not reach here, but satisfy type checker
    if last_exception:
        raise last_exception
    raise RuntimeError("http_get_with_retry exhausted attempts unexpectedly")
