"""
Redis-backed AuthStore.

Stores each AuthRecord as JSON at key: auth:{namespace}:{auth_id}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis

from .core import AuthRecord, AuthStatus, AuthStore, ModelState, QuotaState
from .file_store import _deserialize_auth, _serialize_auth


class RedisAuthStore(AuthStore):
    def __init__(self, redis_client: redis.Redis) -> None:
        self.client = redis_client

    def _key(self, namespace: str, auth_id: str) -> str:
        return f"auth:{namespace}:{auth_id}"

    def get(self, namespace: str, auth_id: str) -> Optional[AuthRecord]:
        raw = self.client.get(self._key(namespace, auth_id))
        if raw is None:
            return None
        data = json.loads(raw)
        return _deserialize_auth(data)

    def save(self, namespace: str, record: AuthRecord) -> AuthRecord:
        record.updated_at = datetime.now(timezone.utc)
        payload = json.dumps(_serialize_auth(record))
        self.client.set(self._key(namespace, record.id), payload)
        return record

    def delete(self, namespace: str, auth_id: str) -> None:
        self.client.delete(self._key(namespace, auth_id))

    def list(self, namespace: str) -> List[AuthRecord]:
        prefix = f"auth:{namespace}:"
        keys = [k for k in self.client.scan_iter(f"{prefix}*")]
        records: List[AuthRecord] = []
        for key in keys:
            raw = self.client.get(key)
            if raw is None:
                continue
            try:
                data = json.loads(raw)
                records.append(_deserialize_auth(data))
            except Exception:
                continue
        return records
