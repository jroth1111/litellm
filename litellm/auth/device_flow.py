"""
Minimal helpers for device/browser OAuth flows using TokenStorage.

This keeps login flows decoupled from AuthStore/Router: a device flow writes a
TokenRecord, and the poller checks for completion or expiry.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from .token_storage import TokenRecord, TokenStorage

_STATUS_PENDING = "pending"
_STATUS_AUTHORIZED = "authorized"
_STATUS_EXPIRED = "expired"


def start_device_flow(
    storage: TokenStorage,
    token_id: str,
    provider: str,
    verification_uri: str,
    user_code: str,
    device_code: str,
    interval_seconds: int,
    expires_in: int,
) -> TokenRecord:
    """
    Initialize a device flow token record with pending status.
    """
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    record = TokenRecord(
        id=token_id,
        provider=provider,
        metadata={
            "status": _STATUS_PENDING,
            "verification_uri": verification_uri,
            "user_code": user_code,
            "device_code": device_code,
            "interval_seconds": interval_seconds,
            "expires_at": expires_at.isoformat(),
        },
    )
    return storage.save(record)


def mark_device_flow_authorized(
    storage: TokenStorage,
    token_id: str,
    access_token: str,
    refresh_token: Optional[str] = None,
    expires_in: Optional[int] = None,
) -> Optional[TokenRecord]:
    rec = storage.get(token_id)
    if rec is None:
        return None
    rec = rec
    rec.metadata["status"] = _STATUS_AUTHORIZED
    rec.metadata["access_token"] = access_token
    if refresh_token:
        rec.metadata["refresh_token"] = refresh_token
    if expires_in:
        rec.metadata["expires_at"] = (
            datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        ).isoformat()
    return storage.save(rec)


def poll_device_flow(
    storage: TokenStorage,
    token_id: str,
    max_attempts: int = 10,
    sleep_func: Optional[Callable[[float], None]] = None,
) -> Optional[TokenRecord]:
    """
    Poll for completion; returns TokenRecord when authorized or None if expired/timeout.
    """
    sleep_fn = sleep_func or time.sleep
    for _ in range(max_attempts):
        rec = storage.get(token_id)
        if rec is None:
            return None
        status = (rec.metadata or {}).get("status", _STATUS_PENDING)
        expires_at = rec.metadata.get("expires_at")
        if expires_at:
            try:
                if datetime.fromisoformat(str(expires_at)) < datetime.now(timezone.utc):
                    rec.metadata["status"] = _STATUS_EXPIRED
                    storage.save(rec)
                    return None
            except Exception:
                pass
        if status == _STATUS_AUTHORIZED and rec.metadata.get("access_token"):
            return rec
        interval = rec.metadata.get("interval_seconds") or 5
        try:
            interval = float(interval)
        except Exception:
            interval = 5.0
        sleep_fn(interval)
    return None
