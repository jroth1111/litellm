"""
Object-store-backed AuthStore (S3-compatible or GCS via boto3-style clients).

This keeps dependencies abstract by requiring a minimal client with
`get_object`/`put_object`/`list_objects`/`delete_object` semantics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional

from .core import AuthRecord, AuthStore
from .file_store import _deserialize_auth, _serialize_auth


class ObjectAuthStore(AuthStore):
    def __init__(self, bucket: str, prefix: str, client) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = client

    def _key(self, namespace: str, auth_id: str) -> str:
        return f"{self.prefix}/{namespace}/{auth_id}.json"

    def get(self, namespace: str, auth_id: str) -> Optional[AuthRecord]:
        key = self._key(namespace, auth_id)
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
            body = obj["Body"].read()
        except Exception:
            return None
        data = json.loads(body)
        return _deserialize_auth(data)

    def save(self, namespace: str, record: AuthRecord) -> AuthRecord:
        record.updated_at = datetime.now(timezone.utc)
        payload = json.dumps(_serialize_auth(record)).encode("utf-8")
        key = self._key(namespace, record.id)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=payload)
        return record

    def delete(self, namespace: str, auth_id: str) -> None:
        key = self._key(namespace, auth_id)
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        except Exception:
            return

    def list(self, namespace: str) -> List[AuthRecord]:
        prefix = f"{self.prefix}/{namespace}/"
        try:
            resp = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        except Exception:
            return []
        contents = resp.get("Contents", []) or []
        records: List[AuthRecord] = []
        for item in contents:
            key = item.get("Key")
            if not key or not key.endswith(".json"):
                continue
            try:
                obj = self.client.get_object(Bucket=self.bucket, Key=key)
                body = obj["Body"].read()
                data = json.loads(body)
                records.append(_deserialize_auth(data))
            except Exception:
                continue
        return records
