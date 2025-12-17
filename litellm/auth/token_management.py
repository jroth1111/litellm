"""
Import/export helpers for TokenStorage (namespaced snapshots).
"""

from __future__ import annotations

import json
from typing import List

from .token_storage import TokenRecord, TokenStorage


def export_tokens(storage: TokenStorage, out_path: str) -> None:
    tokens = storage.list()
    payload = []
    for t in tokens:
        entry = {
            "id": t.id,
            "provider": t.provider,
            "scopes": t.scopes,
            "expires_at": t.expires_at.isoformat() if t.expires_at else None,
            "metadata": t.metadata,
            "created_at": t.created_at.isoformat(),
            "updated_at": t.updated_at.isoformat(),
        }
        payload.append(entry)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def import_tokens(storage: TokenStorage, in_path: str) -> List[TokenRecord]:
    with open(in_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    imported: List[TokenRecord] = []
    if not isinstance(payload, list):
        return imported
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        token = TokenRecord(
            id=entry["id"],
            provider=entry["provider"],
            scopes=entry.get("scopes", []) or [],
            expires_at=None,
            metadata=entry.get("metadata", {}) or {},
        )
        storage.save(token)
        imported.append(token)
    return imported
