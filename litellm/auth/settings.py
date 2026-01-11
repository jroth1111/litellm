"""
Validated configuration for subscription OAuth auth storage in the LiteLLM proxy.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import AliasChoices, BaseModel, Field
from pydantic.config import ConfigDict
from pydantic.functional_validators import field_validator

from .crypto import validate_auth_encryption_key
from .paths import default_auth_store_dir


class AuthSettings(BaseModel):
    """
    `auth_settings` block from proxy config.

    Breaking defaults vs earlier prototype:
    - `mount_api` defaults to False (do not expose token CRUD endpoints by default).
    - `store_backend` defaults to encrypted JSON.
    """

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True

    store_dir: str = Field(
        default_factory=default_auth_store_dir,
        validation_alias=AliasChoices("store_dir", "store"),
    )
    namespace: str = Field(default="default", validation_alias=AliasChoices("namespace", "ns"))

    mount_api: bool = False
    maintainer: bool = False
    maintainer_interval_seconds: int = 60

    store_backend: Literal["encrypted_json", "json"] = Field(
        default="encrypted_json",
        validation_alias=AliasChoices("store_backend", "backend"),
    )
    encryption_key: Optional[str] = None
    preferred_alg: Optional[Literal["aesgcm", "fernet", "secretbox"]] = None
    allow_plaintext_fallback: bool = False
    adapter_paths: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("adapter_paths", "adapters"),
    )

    @field_validator("namespace")
    @classmethod
    def _namespace_non_empty(cls, v: str) -> str:
        v2 = (v or "").strip()
        if not v2:
            raise ValueError("namespace must be non-empty")
        return v2

    @field_validator("maintainer_interval_seconds")
    @classmethod
    def _interval_sane(cls, v: int) -> int:
        try:
            v2 = int(v)
        except Exception as e:
            raise ValueError("maintainer_interval_seconds must be an int") from e
        if v2 < 10:
            raise ValueError("maintainer_interval_seconds must be >= 10")
        return v2

    @field_validator("adapter_paths")
    @classmethod
    def _adapter_paths(cls, v: List[str]) -> List[str]:
        if not v:
            return []
        cleaned: List[str] = []
        for item in v:
            if isinstance(item, str) and item.strip():
                cleaned.append(item.strip())
        return cleaned

    @field_validator("encryption_key")
    @classmethod
    def _validate_encryption_key(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return validate_auth_encryption_key(v)
