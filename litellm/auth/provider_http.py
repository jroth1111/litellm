"""
Shared HTTP helpers for provider OAuth flows with minimal retry/backoff.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("litellm.auth.oauth")
if os.getenv("LITELLM_OAUTH_DEBUG"):
    logger.setLevel(logging.DEBUG)


def http_post_with_retry(
    url: str,
    *,
    data: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    max_attempts: int = 3,
    backoff_seconds: float = 0.5,
    retry_on_status: tuple = (429, 500, 502, 503, 504),
) -> httpx.Response:
    """
    Minimal retry helper for token/userinfo requests.
    Retries on a small set of transient status codes.
    """
    attempt = 0
    while True:
        attempt += 1
        start = time.monotonic()
        with httpx.Client(timeout=timeout) as client:
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
        time.sleep(backoff_seconds * attempt)
