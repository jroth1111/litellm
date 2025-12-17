"""
Validated configuration for subscription OAuth auth storage in the LiteLLM proxy.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, Field
from pydantic.config import ConfigDict
from pydantic.functional_validators import field_validator

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
    preferred_alg: Optional[Literal["fernet", "secretbox"]] = None
    allow_plaintext_fallback: bool = False

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
