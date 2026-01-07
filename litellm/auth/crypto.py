"""
Authenticated encryption helpers for persisting OAuth subscription tokens.

Design goals:
- No new mandatory dependencies: prefer `cryptography` when present, fall back to
  `pynacl` when present, otherwise fail loudly.
- Secure key derivation using PBKDF2-HMAC-SHA256 (OWASP 2023 recommendations).
- Self-describing on-disk wrapper so we can change algorithms later.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Literal, Optional, Protocol

__all__ = [
    "AuthEncryptionError",
    "resolve_auth_encryption_secret",
    "build_encryptor",
    "encrypt_bytes",
    "decrypt_bytes",
]

_logger = logging.getLogger(__name__)

# PBKDF2 parameters per OWASP 2023 recommendations
# https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html
_KDF_ITERATIONS = 600_000
_KDF_SALT = b"litellm-auth-v2"  # Static salt for deterministic derivation


class AuthEncryptionError(RuntimeError):
    pass


def resolve_auth_encryption_secret(explicit: Optional[str] = None) -> str:
    """
    Resolve the operator-provided secret used to derive an encryption key.

    Order:
    1) explicit argument
    2) LITELLM_AUTH_ENCRYPTION_KEY
    3) LITELLM_MASTER_KEY
    """
    if explicit:
        return explicit
    env = os.getenv("LITELLM_AUTH_ENCRYPTION_KEY")
    if env:
        return env
    env = os.getenv("LITELLM_MASTER_KEY")
    if env:
        return env
    raise AuthEncryptionError(
        "Missing auth encryption secret. Set `LITELLM_AUTH_ENCRYPTION_KEY` (preferred) "
        "or `LITELLM_MASTER_KEY`."
    )


def _derive_key_bytes(secret: str) -> bytes:
    """
    Derive 32 bytes from an arbitrary secret string using PBKDF2-HMAC-SHA256.

    Uses OWASP 2023 recommended iteration count (600,000) for security against
    brute-force attacks on weak secrets.
    """
    return hashlib.pbkdf2_hmac(
        "sha256",
        secret.encode("utf-8"),
        _KDF_SALT,
        _KDF_ITERATIONS,
        dklen=32,
    )


class _Encryptor(Protocol):
    alg: str

    def encrypt(self, plaintext: bytes) -> bytes: ...
    def decrypt(self, ciphertext: bytes) -> bytes: ...


def _build_fernet_encryptor(secret: str) -> _Encryptor:
    try:
        from cryptography.fernet import Fernet
    except Exception as e:  # pragma: no cover
        raise AuthEncryptionError(
            "cryptography is required for fernet encryption but is not available"
        ) from e

    key = base64.urlsafe_b64encode(_derive_key_bytes(secret))
    f = Fernet(key)

    class _FernetEncryptor:
        alg = "fernet"

        def encrypt(self, plaintext: bytes) -> bytes:
            return f.encrypt(plaintext)

        def decrypt(self, ciphertext: bytes) -> bytes:
            return f.decrypt(ciphertext)

    return _FernetEncryptor()


def _build_secretbox_encryptor(secret: str) -> _Encryptor:
    try:
        from nacl.secret import SecretBox
        import nacl.utils
    except Exception as e:  # pragma: no cover
        raise AuthEncryptionError(
            "pynacl is required for secretbox encryption but is not available"
        ) from e

    key = _derive_key_bytes(secret)
    box = SecretBox(key)

    class _SecretBoxEncryptor:
        alg = "secretbox"

        def encrypt(self, plaintext: bytes) -> bytes:
            nonce = nacl.utils.random(SecretBox.NONCE_SIZE)
            return bytes(box.encrypt(plaintext, nonce))

        def decrypt(self, ciphertext: bytes) -> bytes:
            return box.decrypt(ciphertext)

    return _SecretBoxEncryptor()


def build_encryptor(
    secret: str,
    preferred: Optional[Literal["fernet", "secretbox"]] = None,
) -> _Encryptor:
    """
    Choose an encryptor implementation based on availability and preference.
    """
    if preferred == "fernet":
        return _build_fernet_encryptor(secret)
    if preferred == "secretbox":
        return _build_secretbox_encryptor(secret)

    # Prefer Fernet when available.
    try:
        return _build_fernet_encryptor(secret)
    except AuthEncryptionError:
        return _build_secretbox_encryptor(secret)


@dataclass(frozen=True)
class EncryptedEnvelope:
    v: int
    alg: str
    ct: str


def encrypt_bytes(plaintext: bytes, *, encryptor: _Encryptor) -> bytes:
    token = encryptor.encrypt(plaintext)
    if encryptor.alg == "fernet":
        ct = token.decode("utf-8")
    else:
        ct = base64.urlsafe_b64encode(token).decode("utf-8")
    env = EncryptedEnvelope(v=1, alg=encryptor.alg, ct=ct)
    return json.dumps(env.__dict__, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def decrypt_bytes(ciphertext: bytes, *, secret: str) -> bytes:
    """Decrypt an encrypted auth record."""
    try:
        env = json.loads(ciphertext.decode("utf-8"))
    except Exception as e:
        raise AuthEncryptionError("Invalid encrypted auth record: not valid JSON") from e
    if not isinstance(env, dict):
        raise AuthEncryptionError("Invalid encrypted auth record: expected object")
    alg = str(env.get("alg") or "").strip().lower()
    ct = env.get("ct")
    if not alg or not isinstance(ct, str) or not ct:
        raise AuthEncryptionError("Invalid encrypted auth record: missing fields")

    encryptor = build_encryptor(
        secret, preferred=alg if alg in ("fernet", "secretbox") else None
    )  # type: ignore[arg-type]
    if encryptor.alg == "fernet":
        token = ct.encode("utf-8")
    else:
        token = base64.urlsafe_b64decode(ct.encode("utf-8"))
    try:
        return encryptor.decrypt(token)
    except Exception as e:
        raise AuthEncryptionError("Failed to decrypt auth record (wrong key?)") from e
