"""
Redis-backed AuthStore.

Stores each AuthRecord as JSON at key: auth:{namespace}:{auth_id}
Provides distributed locking via Redis SET NX for multi-instance deployments.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

import redis

from .core import AuthRecord, AuthStatus, AuthStore, ModelState, QuotaState
from .file_store import _deserialize_auth, _serialize_auth


class RedisAuthStore(AuthStore):
    def __init__(
        self, redis_client: redis.Redis, lock_ttl_seconds: int = 30
    ) -> None:
        self.client = redis_client
        self.lock_ttl = lock_ttl_seconds

    def _key(self, namespace: str, auth_id: str) -> str:
        return f"auth:{namespace}:{auth_id}"

    def _lock_key(self, namespace: str, auth_id: str) -> str:
        return f"auth_lock:{namespace}:{auth_id}"

    @contextmanager
    def lock(
        self,
        namespace: str,
        auth_id: str,
        *,
        timeout_seconds: float = 10.0,
    ) -> Iterator[bool]:
        """
        Distributed lock for auth record operations using Redis SET NX.

        Usage:
            with store.lock(namespace, auth_id) as acquired:
                if acquired:
                    # perform protected operation
                    record = store.get(namespace, auth_id)
                    # ... modify ...
                    store.save(namespace, record)

        Args:
            namespace: Auth namespace.
            auth_id: Auth record ID to lock.
            timeout_seconds: Max time to wait for lock acquisition.

        Yields:
            True if lock was acquired, False if timeout reached.
        """
        lock_key = self._lock_key(namespace, auth_id)
        # Use UUID to ensure we only release our own lock
        lock_value = str(uuid.uuid4())
        deadline = time.time() + max(0.1, float(timeout_seconds))
        acquired = False

        try:
            while time.time() < deadline:
                # SET NX with TTL for distributed lock
                if self.client.set(lock_key, lock_value, nx=True, ex=self.lock_ttl):
                    acquired = True
                    break
                time.sleep(0.05)
            yield acquired
        finally:
            if acquired:
                # Only release if we still own the lock (compare value)
                # Use Lua script for atomic check-and-delete
                release_script = """
                if redis.call("get", KEYS[1]) == ARGV[1] then
                    return redis.call("del", KEYS[1])
                else
                    return 0
                end
                """
                self.client.eval(release_script, 1, lock_key, lock_value)

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
