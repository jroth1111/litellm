"""
PKCE (Proof Key for Code Exchange) utilities for OAuth 2.0 flows.

Implements RFC 7636 for enhanced security in public clients.
Used by providers that require PKCE (e.g., Qwen).
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from typing import Tuple


def generate_code_verifier(length: int = 32) -> str:
    """
    Generate a cryptographically random code verifier for PKCE.

    Args:
        length: Number of random bytes to use (default 32, produces 43-char string).

    Returns:
        URL-safe base64-encoded string without padding.
    """
    random_bytes = secrets.token_bytes(length)
    return base64.urlsafe_b64encode(random_bytes).rstrip(b"=").decode("ascii")


def generate_code_challenge(code_verifier: str) -> str:
    """
    Generate a SHA-256 code challenge from the code verifier.

    Args:
        code_verifier: The code verifier string.

    Returns:
        URL-safe base64-encoded SHA-256 hash without padding.
    """
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def generate_pkce_pair() -> Tuple[str, str]:
    """
    Generate a complete PKCE pair (code_verifier, code_challenge).

    Returns:
        Tuple of (code_verifier, code_challenge) strings.
    """
    verifier = generate_code_verifier()
    challenge = generate_code_challenge(verifier)
    return verifier, challenge


def generate_state(length: int = 16) -> str:
    """
    Generate a cryptographically random state parameter for CSRF protection.

    Args:
        length: Number of random bytes to use (default 16).

    Returns:
        URL-safe base64-encoded string without padding.
    """
    random_bytes = secrets.token_bytes(length)
    return base64.urlsafe_b64encode(random_bytes).rstrip(b"=").decode("ascii")
