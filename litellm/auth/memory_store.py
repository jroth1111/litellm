"""
In-memory AuthStore (non-durable) for tests and short-lived scenarios.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .core import AuthRecord, AuthStore


class InMemoryAuthStore(AuthStore):
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, AuthRecord]] = {}

    def _ns(self, namespace: str) -> Dict[str, AuthRecord]:
        return self._data.setdefault(namespace, {})

    def get(self, namespace: str, auth_id: str) -> Optional[AuthRecord]:
        return self._ns(namespace).get(auth_id)

    def save(self, namespace: str, record: AuthRecord) -> AuthRecord:
        self._ns(namespace)[record.id] = record
        return record

    def delete(self, namespace: str, auth_id: str) -> None:
        self._ns(namespace).pop(auth_id, None)

    def list(self, namespace: str) -> List[AuthRecord]:
        return list(self._ns(namespace).values())
