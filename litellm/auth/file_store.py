"""
JSON file-based AuthStore implementation.

Each namespace gets its own subdirectory under `base_dir`. Auth records are
persisted as `<auth_id>.json` with ISO 8601 timestamps.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .core import AuthRecord, AuthStatus, AuthStore, ModelState, QuotaState
from .crypto import (
    AuthEncryptionError,
    build_encryptor,
    decrypt_bytes,
    encrypt_bytes,
    resolve_auth_encryption_secret,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "AuthLoadError",
    "JsonFileAuthStore",
    "EncryptedJsonFileAuthStore",
]


@dataclass(frozen=True)
class AuthLoadError:
    path: str
    error_type: str
    message: str


_TOKEN_KEYS: Tuple[str, ...] = (
    "access_token",
    "refresh_token",
    "expires_at",
    "id_token",
    "token_type",
)


@contextmanager
def _exclusive_file_lock(lock_path: str, *, timeout_seconds: float = 10.0) -> Iterator[None]:
    """
    Cross-process best-effort file lock.

    Uses `fcntl.flock` on POSIX and `msvcrt.locking` on Windows.
    """
    _ensure_private_dir(os.path.dirname(lock_path))
    f = open(lock_path, "a+b")
    try:
        try:
            os.chmod(lock_path, 0o600)
        except Exception:
            pass

        deadline = time.time() + max(0.1, float(timeout_seconds))
        # POSIX
        try:
            import fcntl  # type: ignore

            while True:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() >= deadline:
                        raise TimeoutError(f"timed out acquiring file lock: {lock_path}")
                    time.sleep(0.05)
        except Exception:
            # Windows / other
            try:
                import msvcrt  # type: ignore

                while True:
                    try:
                        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.time() >= deadline:
                            raise TimeoutError(f"timed out acquiring file lock: {lock_path}")
                        time.sleep(0.05)
            except Exception:
                # No usable locking backend; proceed without lock.
                _logger.warning(
                    "No file locking available; concurrent writes may cause data loss: %s",
                    lock_path,
                )

        yield
    finally:
        # Best-effort unlock
        try:
            import fcntl  # type: ignore

            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            try:
                import msvcrt  # type: ignore

                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
        try:
            f.close()
        except Exception:
            pass


def _serialize_datetime(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _deserialize_datetime(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _serialize_model_state(state: ModelState) -> Dict[str, Any]:
    return {
        "status": state.status.value,
        "status_message": state.status_message,
        "unavailable": state.unavailable,
        "next_retry_after": _serialize_datetime(state.next_retry_after),
        "last_error": state.last_error,
        "quota": {
            "exceeded": state.quota.exceeded,
            "reason": state.quota.reason,
            "next_recover_at": _serialize_datetime(state.quota.next_recover_at),
            "backoff_level": state.quota.backoff_level,
        },
        "updated_at": _serialize_datetime(state.updated_at),
    }


def _deserialize_model_state(data: Dict[str, Any]) -> ModelState:
    quota = data.get("quota", {}) or {}
    return ModelState(
        status=AuthStatus(data.get("status", AuthStatus.ACTIVE)),
        status_message=data.get("status_message", ""),
        unavailable=bool(data.get("unavailable", False)),
        next_retry_after=_deserialize_datetime(data.get("next_retry_after")),
        last_error=data.get("last_error"),
        quota=QuotaState(
            exceeded=bool(quota.get("exceeded", False)),
            reason=quota.get("reason", ""),
            next_recover_at=_deserialize_datetime(quota.get("next_recover_at")),
            backoff_level=int(quota.get("backoff_level", 0)),
        ),
        updated_at=_deserialize_datetime(data.get("updated_at"))
        or datetime.now(timezone.utc),
    )


def _serialize_auth(record: AuthRecord) -> Dict[str, Any]:
    return {
        "id": record.id,
        "provider": record.provider,
        "label": record.label,
        "attributes": record.attributes,
        "metadata": record.metadata,
        "status": record.status.value,
        "status_message": record.status_message,
        "unavailable": record.unavailable,
        "quota": {
            "exceeded": record.quota.exceeded,
            "reason": record.quota.reason,
            "next_recover_at": _serialize_datetime(record.quota.next_recover_at),
            "backoff_level": record.quota.backoff_level,
        },
        "model_states": {
            key: _serialize_model_state(state)
            for key, state in record.model_states.items()
        },
        "created_at": _serialize_datetime(record.created_at),
        "updated_at": _serialize_datetime(record.updated_at),
        "last_refreshed_at": _serialize_datetime(record.last_refreshed_at),
        "next_refresh_after": _serialize_datetime(record.next_refresh_after),
        "next_retry_after": _serialize_datetime(record.next_retry_after),
        "request_count": int(record.request_count),
        "error_count": int(record.error_count),
        "prompt_tokens": int(record.prompt_tokens),
        "completion_tokens": int(record.completion_tokens),
        "last_request_at": _serialize_datetime(record.last_request_at),
    }


def _deserialize_auth(data: Dict[str, Any]) -> AuthRecord:
    quota = data.get("quota", {}) or {}
    model_states_raw = data.get("model_states", {}) or {}
    model_states = {
        key: _deserialize_model_state(value) for key, value in model_states_raw.items()
    }
    return AuthRecord(
        id=data["id"],
        provider=data["provider"],
        label=data.get("label", ""),
        attributes=data.get("attributes", {}) or {},
        metadata=data.get("metadata", {}) or {},
        status=AuthStatus(data.get("status", AuthStatus.ACTIVE)),
        status_message=data.get("status_message", ""),
        unavailable=bool(data.get("unavailable", False)),
        quota=QuotaState(
            exceeded=bool(quota.get("exceeded", False)),
            reason=quota.get("reason", ""),
            next_recover_at=_deserialize_datetime(quota.get("next_recover_at")),
            backoff_level=int(quota.get("backoff_level", 0)),
        ),
        model_states=model_states,
        created_at=_deserialize_datetime(data.get("created_at"))
        or datetime.now(timezone.utc),
        updated_at=_deserialize_datetime(data.get("updated_at"))
        or datetime.now(timezone.utc),
        last_refreshed_at=_deserialize_datetime(data.get("last_refreshed_at")),
        next_refresh_after=_deserialize_datetime(data.get("next_refresh_after")),
        next_retry_after=_deserialize_datetime(data.get("next_retry_after")),
        request_count=int(data.get("request_count", 0) or 0),
        error_count=int(data.get("error_count", 0) or 0),
        prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
        completion_tokens=int(data.get("completion_tokens", 0) or 0),
        last_request_at=_deserialize_datetime(data.get("last_request_at")),
    )


def _ensure_private_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except Exception:
        # best-effort on non-POSIX filesystems
        pass


def _atomic_write_bytes(path: str, data: bytes, *, mode: int = 0o600) -> None:
    """
    Atomic write: write to a temp file in the same directory, fsync, then replace.
    Ensures secrets are not written partially on crash/kill.
    """
    dir_path = os.path.dirname(path)
    _ensure_private_dir(dir_path)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp.", dir=dir_path)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, path)
        try:
            os.chmod(path, mode)
        except Exception:
            pass
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _merge_auth_records(existing: AuthRecord, incoming: AuthRecord) -> AuthRecord:
    """
    Merge two records to reduce lost updates under multi-process writers.

    - Model states are unioned (per-model newest `updated_at` wins).
    - Token-like metadata keys are protected by `last_refreshed_at` (newer wins).
    - Incoming status/quota fields win (latest call semantics).
    """
    merged = incoming.clone()

    # Preserve original creation time.
    merged.created_at = existing.created_at

    # Merge immutable-ish provider config.
    merged.attributes = dict(existing.attributes or {})
    merged.attributes.update(incoming.attributes or {})

    # Merge model states (preserve states not present in incoming).
    merged.model_states = dict(existing.model_states or {})
    for key, state_in in (incoming.model_states or {}).items():
        prev = merged.model_states.get(key)
        if prev is None or state_in.updated_at >= prev.updated_at:
            merged.model_states[key] = state_in

    # Merge metadata. Protect tokens by last_refreshed_at.
    merged.metadata = dict(existing.metadata or {})
    merged.metadata.update(incoming.metadata or {})

    existing_refreshed = existing.last_refreshed_at or datetime.min.replace(
        tzinfo=timezone.utc
    )
    incoming_refreshed = incoming.last_refreshed_at or datetime.min.replace(
        tzinfo=timezone.utc
    )
    if existing_refreshed > incoming_refreshed:
        for key in _TOKEN_KEYS:
            if key in (existing.metadata or {}):
                merged.metadata[key] = existing.metadata[key]
        merged.last_refreshed_at = existing.last_refreshed_at
    else:
        merged.last_refreshed_at = incoming.last_refreshed_at or existing.last_refreshed_at

    return merged


class JsonFileAuthStore(AuthStore):
    """
    Minimal JSON-file-backed AuthStore.
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir = os.path.abspath(base_dir)
        self.last_load_errors: List[AuthLoadError] = []
        _ensure_private_dir(self.base_dir)

    def _ns_dir(self, namespace: str) -> str:
        path = os.path.join(self.base_dir, namespace)
        _ensure_private_dir(path)
        return path

    def _path(self, namespace: str, auth_id: str) -> str:
        return os.path.join(self._ns_dir(namespace), f"{auth_id}.json")

    def _lock_path(self, namespace: str, auth_id: str) -> str:
        return self._path(namespace, auth_id) + ".lock"

    def get(self, namespace: str, auth_id: str) -> Optional[AuthRecord]:
        path = self._path(namespace, auth_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _deserialize_auth(data)

    def save(self, namespace: str, record: AuthRecord) -> AuthRecord:
        path = self._path(namespace, record.id)
        lock_path = self._lock_path(namespace, record.id)
        with _exclusive_file_lock(lock_path):
            existing: Optional[AuthRecord] = None
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    existing = _deserialize_auth(data)
                except Exception as e:
                    _logger.warning(
                        "Failed to read existing auth record for merge (%s): %s", path, e
                    )
                    existing = None

            to_save = _merge_auth_records(existing, record) if existing else record.clone()
            to_save.updated_at = datetime.now(timezone.utc)
            payload = json.dumps(
                _serialize_auth(to_save), ensure_ascii=False, indent=2
            ).encode("utf-8")
            _atomic_write_bytes(path, payload, mode=0o600)
            return to_save

    def delete(self, namespace: str, auth_id: str) -> None:
        path = self._path(namespace, auth_id)
        lock_path = self._lock_path(namespace, auth_id)
        with _exclusive_file_lock(lock_path):
            if os.path.exists(path):
                os.remove(path)

    def list(self, namespace: str) -> List[AuthRecord]:
        dir_path = self._ns_dir(namespace)
        records: List[AuthRecord] = []
        errors: List[AuthLoadError] = []
        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(dir_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                records.append(_deserialize_auth(data))
            except Exception as e:
                errors.append(
                    AuthLoadError(
                        path=fpath,
                        error_type="json_decode_error",
                        message=str(e),
                    )
                )
                _logger.warning("Skipping malformed auth record file: %s", fpath)
                continue
        self.last_load_errors = errors
        return records

    # convenience helpers for import/export
    def export_to_file(self, namespace: str, out_path: str) -> None:
        records = self.list(namespace)
        payload = [_serialize_auth(rec) for rec in records]
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        _atomic_write_bytes(os.path.abspath(out_path), data, mode=0o600)

    def import_from_file(self, namespace: str, in_path: str) -> List[AuthRecord]:
        with open(in_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError("import payload must be a list")
        imported: List[AuthRecord] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            record = _deserialize_auth(entry)
            self.save(namespace, record)
            imported.append(record)
        return imported


class EncryptedJsonFileAuthStore(AuthStore):
    """
    AuthStore that encrypts AuthRecord JSON at rest.

    On-disk format: a small JSON envelope with ciphertext.

    Secret resolution:
      - explicit `secret` argument
      - env `LITELLM_AUTH_ENCRYPTION_KEY`
      - env `LITELLM_MASTER_KEY`
    """

    def __init__(
        self,
        base_dir: str,
        *,
        secret: Optional[str] = None,
        preferred_alg: Optional[str] = None,
        allow_plaintext_fallback: bool = False,
    ) -> None:
        self.base_dir = os.path.abspath(base_dir)
        self.secret = resolve_auth_encryption_secret(secret)
        self.encryptor = build_encryptor(self.secret, preferred=preferred_alg)  # type: ignore[arg-type]
        self.allow_plaintext_fallback = allow_plaintext_fallback
        self.last_load_errors: List[AuthLoadError] = []
        _ensure_private_dir(self.base_dir)

    def _ns_dir(self, namespace: str) -> str:
        path = os.path.join(self.base_dir, namespace)
        _ensure_private_dir(path)
        return path

    def _path(self, namespace: str, auth_id: str) -> str:
        return os.path.join(self._ns_dir(namespace), f"{auth_id}.json")

    def _lock_path(self, namespace: str, auth_id: str) -> str:
        return self._path(namespace, auth_id) + ".lock"

    def _read_record_bytes(self, path: str) -> Optional[bytes]:
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    def _decode_record(self, raw: bytes) -> AuthRecord:
        try:
            plaintext = decrypt_bytes(raw, secret=self.secret)
            data = json.loads(plaintext.decode("utf-8"))
            if not isinstance(data, dict):
                raise AuthEncryptionError("decrypted payload is not an object")
            return _deserialize_auth(data)
        except AuthEncryptionError:
            if not self.allow_plaintext_fallback:
                raise
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("plaintext payload is not an object")
            return _deserialize_auth(data)

    def get(self, namespace: str, auth_id: str) -> Optional[AuthRecord]:
        path = self._path(namespace, auth_id)
        raw = self._read_record_bytes(path)
        if raw is None:
            return None
        return self._decode_record(raw)

    def save(self, namespace: str, record: AuthRecord) -> AuthRecord:
        path = self._path(namespace, record.id)
        lock_path = self._lock_path(namespace, record.id)
        with _exclusive_file_lock(lock_path):
            existing: Optional[AuthRecord] = None
            raw_existing = self._read_record_bytes(path)
            if raw_existing is not None:
                try:
                    existing = self._decode_record(raw_existing)
                except Exception as e:
                    _logger.warning(
                        "Failed to read existing auth record for merge (%s): %s", path, e
                    )
                    existing = None

            to_save = _merge_auth_records(existing, record) if existing else record.clone()
            to_save.updated_at = datetime.now(timezone.utc)
            plaintext = json.dumps(
                _serialize_auth(to_save), ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            wrapped = encrypt_bytes(plaintext, encryptor=self.encryptor)
            _atomic_write_bytes(path, wrapped, mode=0o600)
            return to_save

    def delete(self, namespace: str, auth_id: str) -> None:
        path = self._path(namespace, auth_id)
        lock_path = self._lock_path(namespace, auth_id)
        with _exclusive_file_lock(lock_path):
            if os.path.exists(path):
                os.remove(path)

    def list(self, namespace: str) -> List[AuthRecord]:
        dir_path = self._ns_dir(namespace)
        records: List[AuthRecord] = []
        errors: List[AuthLoadError] = []
        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(dir_path, fname)
            try:
                raw = self._read_record_bytes(fpath)
                if raw is None:
                    continue
                records.append(self._decode_record(raw))
            except AuthEncryptionError:
                # Wrong key / invalid ciphertext should be loud; otherwise the store
                # can look empty and be very hard to debug.
                raise AuthEncryptionError(f"failed to decrypt auth record file: {fpath}")
            except Exception as e:
                errors.append(
                    AuthLoadError(
                        path=fpath,
                        error_type="decode_error",
                        message=str(e),
                    )
                )
                _logger.warning("Skipping malformed auth record file: %s", fpath)
                continue
        self.last_load_errors = errors
        return records

    def export_to_file(self, namespace: str, out_path: str) -> None:
        records = self.list(namespace)
        payload = [_serialize_auth(rec) for rec in records]
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        _atomic_write_bytes(os.path.abspath(out_path), data, mode=0o600)

    def import_from_file(self, namespace: str, in_path: str) -> List[AuthRecord]:
        with open(in_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError("import payload must be a list")
        imported: List[AuthRecord] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            record = _deserialize_auth(entry)
            self.save(namespace, record)
            imported.append(record)
        return imported
