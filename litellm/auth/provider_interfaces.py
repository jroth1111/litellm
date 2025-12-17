"""
Common interfaces and lightweight DTOs for OAuth-capable providers.

These are intentionally minimal to avoid coupling call sites to any
specific provider while still enabling structured refresh/exchange flows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


@dataclass
class ProviderTokenResult:
    """Standard token payload returned by provider OAuth helpers."""

    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[str] = None  # ISO8601
    token_type: str = "Bearer"
    id_token: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceCodeResult:
    """Device authorization details for device-code providers."""

    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int
    verification_uri_complete: Optional[str] = None
    # Optional PKCE code_verifier (e.g., Qwen uses PKCE with device flow)
    code_verifier: Optional[str] = None


class ProviderOAuth(Protocol):
    """Browser-based OAuth (authorization code) interface."""

    provider: str

    def authorize_url(self, *, state: str, code_challenge: Optional[str] = None, redirect_uri: Optional[str] = None) -> str:
        ...

    def exchange_code(
        self, *, code: str, code_verifier: Optional[str] = None, state: Optional[str] = None, redirect_uri: Optional[str] = None
    ) -> ProviderTokenResult:
        ...

    def refresh(self, *, refresh_token: str) -> ProviderTokenResult:
        ...


class ProviderDeviceOAuth(Protocol):
    """Device-code OAuth interface (e.g., GitHub Copilot, Qwen)."""

    provider: str

    def device_authorize(self) -> DeviceCodeResult:
        ...

    def device_poll(self, *, device_code: DeviceCodeResult) -> ProviderTokenResult:
        ...


class ProviderRequestSigner(Protocol):
    """Injects provider-specific auth headers."""

    def sign(self, headers: Dict[str, str], token: str) -> Dict[str, str]:
        ...


class BearerRequestSigner:
    """Simple Bearer token signer used by most providers."""

    def sign(self, headers: Dict[str, str], token: str) -> Dict[str, str]:
        new_headers = dict(headers)
        new_headers["Authorization"] = f"Bearer {token}"
        return new_headers
