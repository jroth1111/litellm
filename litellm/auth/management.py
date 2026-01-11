"""
Simple management helpers for AuthStore-backed credentials.

This module provides thin utilities for list/download/upload/delete and
import/export of AuthRecords.
"""

from __future__ import annotations

from typing import List, Optional

from .core import AuthRecord, AuthStore
from .file_store import _deserialize_auth, _serialize_auth


def list_auths(store: AuthStore, namespace: str) -> List[AuthRecord]:
    return store.list(namespace)


def get_auth(store: AuthStore, namespace: str, auth_id: str) -> Optional[AuthRecord]:
    return store.get(namespace, auth_id)


def save_auth(store: AuthStore, namespace: str, record: AuthRecord) -> AuthRecord:
    return store.save(namespace, record)


def delete_auth(store: AuthStore, namespace: str, auth_id: str) -> None:
    store.delete(namespace, auth_id)


def export_auths(store: AuthStore, namespace: str, out_path: str) -> None:
    """
    Write all auths in a namespace to a JSON file.
    """
    if hasattr(store, "export_to_file"):
        store.export_to_file(namespace, out_path)  # type: ignore[attr-defined]
        return
    # fallback: manual export
    import json

    records = store.list(namespace)
    payload = [_serialize_auth(record) for record in records]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def import_auths(store: AuthStore, namespace: str, in_path: str) -> List[AuthRecord]:
    """
    Import auths from a JSON file and persist them.
    """
    if hasattr(store, "import_from_file"):
        return store.import_from_file(namespace, in_path)  # type: ignore[attr-defined]
    import json

    with open(in_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    imported: List[AuthRecord] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        record = _deserialize_auth(entry)
        store.save(namespace, record)
        imported.append(record)
    return imported
