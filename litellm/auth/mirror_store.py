"""
Mirror AuthStore that fans out writes to multiple backends and merges reads.

Primary store is authoritative for writes; mirrors are best-effort replicas.
Reads fall back to mirrors and return the newest record by updated_at/created_at.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from .core import AuthRecord, AuthStore

logger = logging.getLogger(__name__)


def _ts(record: AuthRecord) -> datetime:
    if record.updated_at:
        return record.updated_at
    if record.created_at:
        return record.created_at
    return datetime.fromtimestamp(0, tz=timezone.utc)


class MirrorAuthStore(AuthStore):
    """
    Compose multiple stores:
    - Writes go to the primary and then best-effort to mirrors.
    - Reads merge across all stores, preferring the newest record.
    """

    def __init__(self, primary: AuthStore, mirrors: Iterable[AuthStore]) -> None:
        self.primary = primary
        self.mirrors = list(mirrors)

    def get(self, namespace: str, auth_id: str) -> Optional[AuthRecord]:
        record = None
        try:
            record = self.primary.get(namespace, auth_id)
        except Exception as exc:
            logger.debug("mirror store primary get failed: %s", exc)

        newest = record
        for store in self.mirrors:
            try:
                cand = store.get(namespace, auth_id)
            except Exception as exc:
                logger.debug("mirror store mirror get failed: %s", exc)
                continue
            if cand is None:
                continue
            if newest is None or _ts(cand) > _ts(newest):
                newest = cand
        return newest

    def save(self, namespace: str, record: AuthRecord) -> AuthRecord:
        # Ensure updated_at is bumped before replication
        record.updated_at = datetime.now(timezone.utc)
        saved = self.primary.save(namespace, record)
        for store in self.mirrors:
            try:
                store.save(namespace, saved.clone())
            except Exception as exc:
                logger.debug("mirror store mirror save failed: %s", exc)
        return saved

    def delete(self, namespace: str, auth_id: str) -> None:
        try:
            self.primary.delete(namespace, auth_id)
        except Exception as exc:
            logger.debug("mirror store primary delete failed: %s", exc)
        for store in self.mirrors:
            try:
                store.delete(namespace, auth_id)
            except Exception as exc:
                logger.debug("mirror store mirror delete failed: %s", exc)

    def list(self, namespace: str) -> List[AuthRecord]:
        merged: Dict[str, AuthRecord] = {}
        for store in [self.primary, *self.mirrors]:
            try:
                for rec in store.list(namespace):
                    current = merged.get(rec.id)
                    if current is None or _ts(rec) > _ts(current):
                        merged[rec.id] = rec
            except Exception as exc:
                logger.debug("mirror store list failed: %s", exc)
                continue
        return list(merged.values())
