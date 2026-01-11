"""
Authenticated encryption helpers for persisting OAuth subscription tokens.

Design goals:
- AES-256-GCM encryption with per-record nonces.
- Base64-encoded 32-byte key requirement (validated).
- Self-describing on-disk wrapper so we can change algorithms later.
- Backward-compatible decryption for legacy fernet/secretbox payloads.
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
    "generate_auth_encryption_key",
    "validate_auth_encryption_key",
    "build_encryptor",
    "encrypt_bytes",
    "decrypt_bytes",
]

_logger = logging.getLogger(__name__)

# PBKDF2 parameters per OWASP 2023 recommendations (legacy fallback only)
# https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html
_KDF_ITERATIONS = 600_000
_KDF_SALT = b"litellm-auth-v2"  # Static salt for deterministic derivation

_AESGCM_NONCE_BYTES = 12


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
        return validate_auth_encryption_key(explicit)
    env = os.getenv("LITELLM_AUTH_ENCRYPTION_KEY")
    if env:
        return validate_auth_encryption_key(env)
    env = os.getenv("LITELLM_MASTER_KEY")
    if env:
        return validate_auth_encryption_key(env)
    raise AuthEncryptionError(
        "Missing auth encryption secret. Set `LITELLM_AUTH_ENCRYPTION_KEY` (base64 32-byte key, preferred) "
        "or `LITELLM_MASTER_KEY`. Generate one with `litellm auth key init`."
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


def _normalize_b64(value: str) -> str:
    cleaned = (value or "").strip()
    missing = len(cleaned) % 4
    if missing:
        cleaned += "=" * (4 - missing)
    return cleaned


def _decode_base64_key(secret: str) -> bytes:
    try:
        decoded = base64.urlsafe_b64decode(_normalize_b64(secret).encode("utf-8"))
    except Exception as e:
        raise AuthEncryptionError(
            "Invalid auth encryption key. Expected base64-encoded 32-byte key."
        ) from e
    if len(decoded) != 32:
        raise AuthEncryptionError(
            "Invalid auth encryption key length. Expected base64-encoded 32-byte key."
        )
    return decoded


def validate_auth_encryption_key(secret: str) -> str:
    _decode_base64_key(secret)
    return secret.strip()


def generate_auth_encryption_key() -> str:
    """
    Generate a base64-encoded 32-byte key suitable for AES-256-GCM.
    """
    return base64.urlsafe_b64encode(os.urandom(32)).decode("utf-8").rstrip("=")


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


def _build_aesgcm_encryptor(secret: str) -> _Encryptor:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except Exception as e:  # pragma: no cover
        raise AuthEncryptionError(
            "cryptography is required for AES-GCM encryption but is not available"
        ) from e

    key = _decode_base64_key(secret)
    aesgcm = AESGCM(key)

    class _AesGcmEncryptor:
        alg = "aesgcm"

        def encrypt(self, plaintext: bytes) -> bytes:
            nonce = os.urandom(_AESGCM_NONCE_BYTES)
            return nonce + aesgcm.encrypt(nonce, plaintext, None)

        def decrypt(self, ciphertext: bytes) -> bytes:
            if len(ciphertext) < _AESGCM_NONCE_BYTES:
                raise AuthEncryptionError("Invalid AES-GCM payload")
            nonce = ciphertext[:_AESGCM_NONCE_BYTES]
            body = ciphertext[_AESGCM_NONCE_BYTES:]
            return aesgcm.decrypt(nonce, body, None)

    return _AesGcmEncryptor()


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
    preferred: Optional[Literal["aesgcm", "fernet", "secretbox"]] = None,
) -> _Encryptor:
    """
    Choose an encryptor implementation based on availability and preference.
    """
    if preferred == "aesgcm" or preferred is None:
        return _build_aesgcm_encryptor(secret)
    if preferred == "fernet":
        return _build_fernet_encryptor(secret)
    if preferred == "secretbox":
        return _build_secretbox_encryptor(secret)

    return _build_aesgcm_encryptor(secret)


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
    env = EncryptedEnvelope(v=2, alg=encryptor.alg, ct=ct)
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
    if alg not in ("aesgcm", "fernet", "secretbox"):
        raise AuthEncryptionError(f"Unsupported auth encryption algorithm: {alg}")

    encryptor = build_encryptor(
        secret, preferred=alg if alg in ("aesgcm", "fernet", "secretbox") else None
    )
    if encryptor.alg == "fernet":
        token = ct.encode("utf-8")
    else:
        token = base64.urlsafe_b64decode(ct.encode("utf-8"))
    try:
        return encryptor.decrypt(token)
    except Exception as e:
        raise AuthEncryptionError("Failed to decrypt auth record (wrong key?)") from e
