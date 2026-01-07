"""
JWT utilities for OAuth token introspection.

This module provides utilities for parsing JWT tokens without cryptographic
verification. Useful for extracting claims from ID tokens for logging and
observability purposes.

WARNING: Do NOT use these functions for authentication decisions.
Only use for logging/debugging purposes.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, Optional


def parse_jwt_claims(token: str, verify: bool = False) -> Optional[Dict[str, Any]]:
    """
    Parse JWT claims without cryptographic verification.

    Useful for extracting user info from ID tokens for logging/observability.
    NOT for authentication decisions.

    Args:
        token: A JWT token string (header.payload.signature).
        verify: Ignored (included for API compatibility). Verification not supported.

    Returns:
        Parsed claims dictionary, or None if token is invalid.
    """
    if not token or not isinstance(token, str):
        return None

    parts = token.strip().split(".")
    if len(parts) != 3:
        return None

    try:
        # URL-safe base64 decode with padding
        payload_b64 = parts[1]
        # Add padding if needed
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding

        decoded = base64.urlsafe_b64decode(payload_b64)
        return json.loads(decoded.decode("utf-8"))
    except Exception:
        return None


def get_jwt_expiry(token: str) -> Optional[int]:
    """
    Extract 'exp' claim from a JWT token.

    Args:
        token: A JWT token string.

    Returns:
        Unix timestamp of expiry, or None if not present/invalid.
    """
    claims = parse_jwt_claims(token)
    if claims and "exp" in claims:
        try:
            return int(claims["exp"])
        except (ValueError, TypeError):
            return None
    return None


def get_jwt_subject(token: str) -> Optional[str]:
    """
    Extract 'sub' (subject) claim from a JWT token.

    Args:
        token: A JWT token string.

    Returns:
        Subject string, or None if not present/invalid.
    """
    claims = parse_jwt_claims(token)
    if claims and "sub" in claims:
        return str(claims["sub"])
    return None


def get_jwt_email(token: str) -> Optional[str]:
    """
    Extract email claim from a JWT token.

    Checks common claim names: 'email', 'preferred_username', 'upn'.

    Args:
        token: A JWT token string.

    Returns:
        Email string, or None if not found.
    """
    claims = parse_jwt_claims(token)
    if not claims:
        return None

    for key in ("email", "preferred_username", "upn"):
        if key in claims and claims[key]:
            return str(claims[key])
    return None
