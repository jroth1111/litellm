"""
Legacy auth.json migration helpers.

Migrates v1 auth.json (provider->auth info) into the v2 AuthStore layout.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .core import AuthRecord, AuthStatus
from .paths import legacy_auth_json_path

_logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None
    return None


def _infer_kind(meta: Dict[str, Any], attrs: Dict[str, Any]) -> str:
    if meta.get("access_token") or meta.get("refresh_token"):
        return "oauth"
    for key in ("api_key", "apiKey", "key", "token"):
        if key in meta or key in attrs:
            return "api"
    for key in ("env_key", "envKey"):
        if key in meta or key in attrs:
            return "wellknown"
    return "oauth"


def _coerce_record(provider: str, entry: Dict[str, Any]) -> Optional[AuthRecord]:
    provider_key = (provider or "").strip().lower()
    if not provider_key:
        return None

    now = _now()
    meta = dict(entry.get("metadata") or {})
    attrs = dict(entry.get("attributes") or {})

    for key in (
        "access_token",
        "refresh_token",
        "expires_at",
        "id_token",
        "token_type",
        "scope",
    ):
        if key in entry and entry[key] is not None:
            meta.setdefault(key, entry[key])

    if "token" in entry and "access_token" not in meta:
        meta["access_token"] = entry["token"]
    if "api_key" in entry and "api_key" not in meta:
        meta["api_key"] = entry["api_key"]

    label = (
        entry.get("label")
        or entry.get("account")
        or meta.get("account")
        or meta.get("email")
        or ""
    )
    status_raw = entry.get("status") or AuthStatus.ACTIVE
    try:
        status = AuthStatus(status_raw)
    except Exception:
        status = AuthStatus.ACTIVE

    created = _parse_datetime(entry.get("created_at")) or now
    updated = _parse_datetime(entry.get("updated_at")) or now
    auth_id = entry.get("id") or entry.get("auth_id")
    if not auth_id:
        suffix = secrets.token_hex(4)
        ts = now.strftime("%Y%m%d%H%M%S")
        auth_id = f"{provider_key}-legacy-{ts}-{suffix}"

    kind = entry.get("kind") or _infer_kind(meta, attrs)

    return AuthRecord(
        id=str(auth_id),
        provider=provider_key,
        label=str(label or ""),
        attributes={k: str(v) for k, v in attrs.items() if v is not None},
        metadata=meta,
        status=status,
        created_at=created,
        updated_at=updated,
        kind=str(kind),
    )


def _iter_legacy_entries(payload: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(payload, dict):
        for provider, entry in payload.items():
            if isinstance(entry, dict):
                yield str(provider), entry
    elif isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict):
                provider = entry.get("provider") or entry.get("providerId")
                if provider:
                    yield str(provider), entry


def migrate_legacy_auth_json(
    *,
    store,
    namespace: str,
    legacy_path: Optional[str] = None,
) -> List[AuthRecord]:
    """
    Migrate legacy auth.json into the provided AuthStore.
    """
    store_dir = getattr(store, "base_dir", None)
    legacy_path = legacy_path or legacy_auth_json_path(store_dir)
    if not legacy_path or not os.path.exists(legacy_path):
        return []

    try:
        with open(legacy_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        _logger.warning("Legacy auth.json read failed: %s", e)
        return []

    migrated: List[AuthRecord] = []
    for provider, entry in _iter_legacy_entries(payload):
        record = _coerce_record(provider, entry)
        if record is None:
            continue
        existing = store.get(namespace, record.id)
        if existing is not None:
            record.id = f"{record.id}-{secrets.token_hex(2)}"
        store.save(namespace, record)
        migrated.append(record)

    if migrated:
        backup_path = legacy_path + ".bak"
        if os.path.exists(backup_path):
            ts = _now().strftime("%Y%m%d%H%M%S")
            backup_path = legacy_path + f".{ts}.bak"
        try:
            os.replace(legacy_path, backup_path)
        except Exception as e:
            _logger.warning("Legacy auth.json backup failed: %s", e)

    return migrated


def maybe_migrate_legacy_auth_json(
    *,
    store,
    namespace: str,
    legacy_path: Optional[str] = None,
) -> List[AuthRecord]:
    try:
        return migrate_legacy_auth_json(
            store=store, namespace=namespace, legacy_path=legacy_path
        )
    except Exception as e:
        _logger.warning("Legacy auth.json migration failed: %s", e)
        return []
