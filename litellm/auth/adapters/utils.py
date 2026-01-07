from __future__ import annotations

import logging
from typing import Any, Dict

_logger = logging.getLogger(__name__)


def token_dataclass_to_metadata(token: Any) -> Dict[str, Any]:
    """
    Convert provider token dataclass objects into the AuthRecord.metadata shape.

    Providers differ in their payloads; we standardize on keys used by Router:
    - access_token, refresh_token, token_type, expires_at, id_token
    plus any provider-specific identity fields.
    """
    meta: Dict[str, Any] = {}
    if token is None:
        _logger.warning("token_dataclass_to_metadata called with None token")
        return meta


    for key in (
        "access_token",
        "refresh_token",
        "token_type",
        "expires_at",
        "id_token",
    ):
        val = getattr(token, key, None)
        if val is not None:
            meta[key] = val

    for extra in (
        "email",
        "account_id",
        "organization_uuid",
        "organization_name",
        "project_id",
        "resource_url",
        "scope",
        "raw",
        "account",
    ):
        val = getattr(token, extra, None)
        if val is not None:
            meta[extra] = val
    return meta

