"""
Token storage abstraction for login/device flows.

This is separate from AuthStore so login/refresh logic can persist temporary
tokens without coupling to the router's credential manager.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Protocol

from .crypto import (
    AuthEncryptionError,
    build_encryptor,
    decrypt_bytes,
    encrypt_bytes,
    resolve_auth_encryption_secret,
)


def _ensure_private_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except Exception:
        # best-effort on non-POSIX filesystems
        pass


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
                pass

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


def _atomic_write_bytes(path: str, data: bytes, *, mode: int = 0o600) -> None:
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


@dataclass
class TokenRecord:
    id: str
    provider: str
    scopes: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TokenStorage(Protocol):
    def get(self, token_id: str) -> Optional[TokenRecord]:
        ...

    def save(self, record: TokenRecord) -> TokenRecord:
        ...

    def delete(self, token_id: str) -> None:
        ...

    def list(self) -> List[TokenRecord]:
        ...


class PlaintextJsonTokenStorage(TokenStorage):
    """
    Simple file-based token storage for OAuth/device flows.
    Tokens are stored as `<token_id>.json` under base_dir.
    """

    def __init__(self, base_dir: str, namespace: Optional[str] = None) -> None:
        self.base_dir = os.path.abspath(base_dir)
        self.namespace = namespace or "default"
        _ensure_private_dir(self._ns_dir())

    def _ns_dir(self) -> str:
        path = os.path.join(self.base_dir, self.namespace)
        _ensure_private_dir(path)
        return path

    def _path(self, token_id: str) -> str:
        return os.path.join(self._ns_dir(), f"{token_id}.json")

    def _lock_path(self, token_id: str) -> str:
        return self._path(token_id) + ".lock"

    def get(self, token_id: str) -> Optional[TokenRecord]:
        path = self._path(token_id)
        if not os.path.exists(path):
            return None
        with _exclusive_file_lock(self._lock_path(token_id)):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._deserialize(data)

    def save(self, record: TokenRecord) -> TokenRecord:
        record.updated_at = datetime.now(timezone.utc)
        path = self._path(record.id)
        lock_path = self._lock_path(record.id)
        data = json.dumps(
            self._serialize(record), ensure_ascii=False, indent=2
        ).encode("utf-8")
        with _exclusive_file_lock(lock_path):
            _atomic_write_bytes(path, data, mode=0o600)
        return record

    def delete(self, token_id: str) -> None:
        path = self._path(token_id)
        lock_path = self._lock_path(token_id)
        with _exclusive_file_lock(lock_path):
            if os.path.exists(path):
                os.remove(path)

    def list(self) -> List[TokenRecord]:
        records: List[TokenRecord] = []
        ns_dir = self._ns_dir()
        for fname in os.listdir(ns_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(ns_dir, fname)
            try:
                token_id = fname[:-5]
                lock_path = self._lock_path(token_id)
                with _exclusive_file_lock(lock_path):
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    rec = self._deserialize(data)
                    if rec:
                        records.append(rec)
            except Exception:
                continue
        return records

    @staticmethod
    def _serialize(record: TokenRecord) -> Dict[str, Any]:
        return {
            "id": record.id,
            "provider": record.provider,
            "scopes": record.scopes,
            "expires_at": record.expires_at.isoformat() if record.expires_at else None,
            "metadata": record.metadata,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        }

    @staticmethod
    def _deserialize(data: Dict[str, Any]) -> Optional[TokenRecord]:
        try:
            created = datetime.fromisoformat(data.get("created_at"))
        except Exception:
            created = datetime.now(timezone.utc)
        try:
            updated = datetime.fromisoformat(data.get("updated_at"))
        except Exception:
            updated = datetime.now(timezone.utc)
        expires_at = None
        if data.get("expires_at"):
            try:
                expires_at = datetime.fromisoformat(data.get("expires_at"))
            except Exception:
                expires_at = None
        return TokenRecord(
            id=data["id"],
            provider=data["provider"],
            scopes=data.get("scopes", []) or [],
            expires_at=expires_at,
            metadata=data.get("metadata", {}) or {},
            created_at=created,
            updated_at=updated,
        )


class EncryptedJsonTokenStorage(TokenStorage):
    """
    File-based token storage encrypted at rest.

    On-disk format matches `EncryptedJsonFileAuthStore`:
    `{ "v": 1, "alg": "fernet|secretbox", "ct": "<ciphertext>" }`

    Secret resolution:
      - explicit `secret` argument
      - env `LITELLM_AUTH_ENCRYPTION_KEY`
      - env `LITELLM_MASTER_KEY`
    """

    def __init__(
        self,
        base_dir: str,
        namespace: Optional[str] = None,
        *,
        secret: Optional[str] = None,
        preferred_alg: Optional[str] = None,
        allow_plaintext_fallback: bool = True,
    ) -> None:
        self.base_dir = os.path.abspath(base_dir)
        self.namespace = namespace or "default"
        self.secret = resolve_auth_encryption_secret(secret)
        self.encryptor = build_encryptor(self.secret, preferred=preferred_alg)  # type: ignore[arg-type]
        self.allow_plaintext_fallback = allow_plaintext_fallback
        _ensure_private_dir(self._ns_dir())

    def _ns_dir(self) -> str:
        path = os.path.join(self.base_dir, self.namespace)
        _ensure_private_dir(path)
        return path

    def _path(self, token_id: str) -> str:
        return os.path.join(self._ns_dir(), f"{token_id}.json")

    def _lock_path(self, token_id: str) -> str:
        return self._path(token_id) + ".lock"

    def _read_bytes(self, path: str) -> Optional[bytes]:
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    def _decode(self, raw: bytes) -> tuple[TokenRecord, bool]:
        try:
            plaintext = decrypt_bytes(raw, secret=self.secret)
            data = json.loads(plaintext.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("decrypted token payload is not an object")
            rec = PlaintextJsonTokenStorage._deserialize(data)
            if rec is None:
                raise ValueError("decrypted token payload could not be decoded")
            return rec, False
        except AuthEncryptionError:
            if not self.allow_plaintext_fallback:
                raise
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception as e:
                raise AuthEncryptionError("invalid token record (neither encrypted nor JSON)") from e
            if not isinstance(data, dict):
                raise AuthEncryptionError("invalid token record (neither encrypted nor JSON object)")
            # Only treat as plaintext token record if it has the expected shape; otherwise this
            # is likely an encrypted envelope with the wrong key.
            if "id" not in data or "provider" not in data:
                raise AuthEncryptionError("failed to decrypt token record (wrong key?)")
            rec = PlaintextJsonTokenStorage._deserialize(data)
            if rec is None:
                raise AuthEncryptionError("plaintext token payload could not be decoded")
            return rec, True

    def get(self, token_id: str) -> Optional[TokenRecord]:
        path = self._path(token_id)
        lock_path = self._lock_path(token_id)
        with _exclusive_file_lock(lock_path):
            raw = self._read_bytes(path)
            if raw is None:
                return None
            rec, was_plaintext = self._decode(raw)
        # Best-effort migration: if plaintext was read, re-save encrypted.
        if was_plaintext:
            try:
                self.save(rec)
            except Exception:
                pass
        return rec

    def save(self, record: TokenRecord) -> TokenRecord:
        record.updated_at = datetime.now(timezone.utc)
        path = self._path(record.id)
        lock_path = self._lock_path(record.id)
        payload = json.dumps(
            PlaintextJsonTokenStorage._serialize(record), ensure_ascii=False, indent=2
        ).encode("utf-8")
        wrapped = encrypt_bytes(payload, encryptor=self.encryptor)
        with _exclusive_file_lock(lock_path):
            _atomic_write_bytes(path, wrapped, mode=0o600)
        return record

    def delete(self, token_id: str) -> None:
        path = self._path(token_id)
        lock_path = self._lock_path(token_id)
        with _exclusive_file_lock(lock_path):
            if os.path.exists(path):
                os.remove(path)

    def list(self) -> List[TokenRecord]:
        records: List[TokenRecord] = []
        ns_dir = self._ns_dir()
        for fname in os.listdir(ns_dir):
            if not fname.endswith(".json"):
                continue
            token_id = fname[:-5]
            fpath = os.path.join(ns_dir, fname)
            lock_path = self._lock_path(token_id)
            try:
                was_plaintext = False
                with _exclusive_file_lock(lock_path):
                    raw = self._read_bytes(fpath)
                    if raw is None:
                        continue
                    rec, was_plaintext = self._decode(raw)
                records.append(rec)
                if was_plaintext:
                    try:
                        self.save(rec)
                    except Exception:
                        pass
            except Exception:
                continue
        return records


# Breaking hardening: JsonTokenStorage is encrypted at rest by default.
JsonTokenStorage = EncryptedJsonTokenStorage


class InMemoryTokenStorage(TokenStorage):
    """
    Ephemeral token storage for device codes/session tokens.
    """

    def __init__(self) -> None:
        self._data: Dict[str, TokenRecord] = {}

    def get(self, token_id: str) -> Optional[TokenRecord]:
        return self._data.get(token_id)

    def save(self, record: TokenRecord) -> TokenRecord:
        record.updated_at = datetime.now(timezone.utc)
        self._data[record.id] = record
        return record

    def delete(self, token_id: str) -> None:
        self._data.pop(token_id, None)

    def list(self) -> List[TokenRecord]:
        return list(self._data.values())

    @staticmethod
    def _serialize(record: TokenRecord) -> Dict[str, str]:
        return {
            "id": record.id,
            "provider": record.provider,
            "scopes": record.scopes,
            "expires_at": record.expires_at.isoformat() if record.expires_at else None,
            "metadata": record.metadata,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        }

    @staticmethod
    def _deserialize(data: Dict[str, str]) -> Optional[TokenRecord]:
        try:
            created = datetime.fromisoformat(data.get("created_at"))
        except Exception:
            created = datetime.now(timezone.utc)
        try:
            updated = datetime.fromisoformat(data.get("updated_at"))
        except Exception:
            updated = datetime.now(timezone.utc)
        return TokenRecord(
            id=data["id"],
            provider=data["provider"],
            metadata=data.get("metadata", {}) or {},
            created_at=created,
            updated_at=updated,
        )
