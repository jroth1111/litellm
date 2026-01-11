"""
Core auth data structures and interfaces used by subscription OAuth flows.

This layer stays provider-agnostic so strategies and stores can be composed
without coupling to any specific vendor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol

__all__ = [
    "AuthKind",
    "AuthStatus",
    "QuotaState",
    "ModelState",
    "AuthRecord",
    "RequestContext",
    "AuthStore",
    "AuthStrategy",
    "expiration_from_metadata",
    "merge_aliases",
]


class AuthStatus(str, Enum):
    """Lifecycle state for a credential or per-model binding."""

    ACTIVE = "active"
    DISABLED = "disabled"
    EXPIRED = "expired"
    ERROR = "error"


class AuthKind(str, Enum):
    """Credential kind for auth records."""

    OAUTH = "oauth"
    API = "api"
    WELLKNOWN = "wellknown"


@dataclass
class QuotaState:
    """Tracks rate-limit state for a credential or model."""

    exceeded: bool = False
    reason: str = ""
    next_recover_at: Optional[datetime] = None
    backoff_level: int = 0


@dataclass
class ModelState:
    """Per-model health state under a single credential."""

    status: AuthStatus = AuthStatus.ACTIVE
    status_message: str = ""
    unavailable: bool = False
    next_retry_after: Optional[datetime] = None
    last_error: Optional[str] = None
    quota: QuotaState = field(default_factory=QuotaState)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AuthRecord:
    """
    Represents one credential/account that can be used against an upstream provider.

    Attributes:
        id: Stable identifier for the auth record.
        provider: Provider key (e.g., "anthropic", "openai", "gemini").
        label: Human-readable label for logging and observability.
        attributes: Immutable provider configuration (client_id, tenant, etc.).
        metadata: Mutable runtime state (access/refresh tokens, expiry, cookies).
        status: Overall lifecycle status of the credential.
        status_message: Optional reason for the current status.
        unavailable: Indicates temporary unavailability (e.g., quota exceeded).
        quota: Global quota/backoff state for the credential.
        model_states: Per-model health data to avoid flapping entire credential.
        created_at/updated_at: Timestamps for auditing.
        last_refreshed_at/next_refresh_after: Token refresh bookkeeping.
        next_retry_after: Earliest time the credential should be retried.
        request_count/error_count: Usage counters (best-effort; no API calls).
        prompt_tokens/completion_tokens: Aggregate token usage counters (optional).
        last_request_at: Last time this credential was used.
    """

    id: str
    provider: str
    label: str = ""
    kind: AuthKind = AuthKind.OAUTH
    attributes: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: AuthStatus = AuthStatus.ACTIVE
    status_message: str = ""
    unavailable: bool = False
    quota: QuotaState = field(default_factory=QuotaState)
    model_states: Dict[str, ModelState] = field(default_factory=dict)
    runtime: Any = None  # optional runtime-only data, not serialized
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_refreshed_at: Optional[datetime] = None
    next_refresh_after: Optional[datetime] = None
    next_retry_after: Optional[datetime] = None
    request_count: int = 0
    error_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    last_request_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            try:
                self.kind = AuthKind(self.kind)
            except Exception:
                self.kind = AuthKind.OAUTH

    def clone(self) -> "AuthRecord":
        """Shallow copy with isolated mutable fields."""
        return AuthRecord(
            id=self.id,
            provider=self.provider,
            label=self.label,
            kind=self.kind,
            attributes=dict(self.attributes),
            metadata=dict(self.metadata),
            status=self.status,
            status_message=self.status_message,
            unavailable=self.unavailable,
            quota=QuotaState(
                exceeded=self.quota.exceeded,
                reason=self.quota.reason,
                next_recover_at=self.quota.next_recover_at,
                backoff_level=self.quota.backoff_level,
            ),
            model_states={
                key: ModelState(
                    status=value.status,
                    status_message=value.status_message,
                    unavailable=value.unavailable,
                    next_retry_after=value.next_retry_after,
                    last_error=value.last_error,
                    quota=QuotaState(
                        exceeded=value.quota.exceeded,
                        reason=value.quota.reason,
                        next_recover_at=value.quota.next_recover_at,
                        backoff_level=value.quota.backoff_level,
                    ),
                    updated_at=value.updated_at,
                )
                for key, value in self.model_states.items()
            },
            created_at=self.created_at,
            updated_at=self.updated_at,
            last_refreshed_at=self.last_refreshed_at,
            next_refresh_after=self.next_refresh_after,
            next_retry_after=self.next_retry_after,
            request_count=self.request_count,
            error_count=self.error_count,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            last_request_at=self.last_request_at,
            runtime=self.runtime,
        )

    def expiration_time(self) -> Optional[datetime]:
        """
        Attempt to derive expiry from metadata using common keys:
        expired, expire, expires_at, expiresAt, expiry, expires, or nested token.
        """
        return expiration_from_metadata(self.metadata)

    def account_identity(self) -> tuple[str, str]:
        """
        Best-effort identity for logging/observability without leaking secrets.

        Returns (kind, identifier):
          - ("oauth", email/project) if metadata carries email/project_id
          - ("api_key", obfuscated) if attributes contain api_key-like entries
          - ("", "") otherwise
        """
        kind = (
            self.kind.value
            if isinstance(self.kind, AuthKind)
            else str(self.kind or AuthKind.OAUTH.value)
        ).lower()
        meta = self.metadata or {}
        attrs = self.attributes or {}
        email = meta.get("email") or meta.get("user")
        project = meta.get("project_id") or meta.get("tenant") or meta.get("workspace")
        email_str = str(email).strip() if email else ""
        project_str = str(project).strip() if project else ""
        if kind == AuthKind.OAUTH.value and email_str and project_str:
            return "oauth", f"{email_str} ({project_str})"
        if kind == AuthKind.OAUTH.value and email_str:
            return "oauth", email_str
        if kind == AuthKind.WELLKNOWN.value:
            env_key = (
                meta.get("env_key")
                or meta.get("envKey")
                or attrs.get("env_key")
                or attrs.get("envKey")
            )
            if env_key:
                return "wellknown", str(env_key)
            return "wellknown", ""
        # fallback to api key-ish attribute names
        for key in ("api_key", "key", "token"):
            if key in attrs and attrs[key]:
                val = str(attrs[key])
                if len(val) > 6:
                    obf = f"{val[:3]}***{val[-2:]}"
                else:
                    obf = "***"
                return "api", obf
        return "", ""

    def resolve_secret(self) -> Optional[str]:
        """
        Resolve the bearer/API secret for this auth record without logging it.
        """
        meta = self.metadata or {}
        attrs = self.attributes or {}
        kind = (
            self.kind.value
            if isinstance(self.kind, AuthKind)
            else str(self.kind or AuthKind.OAUTH.value)
        ).lower()
        if kind == AuthKind.OAUTH.value:
            return meta.get("access_token") or meta.get("token")
        if kind == AuthKind.API.value:
            return (
                meta.get("api_key")
                or meta.get("key")
                or meta.get("token")
                or attrs.get("api_key")
                or attrs.get("key")
                or attrs.get("token")
            )
        if kind == AuthKind.WELLKNOWN.value:
            token = (
                meta.get("token")
                or meta.get("api_key")
                or meta.get("access_token")
                or attrs.get("token")
                or attrs.get("api_key")
                or attrs.get("access_token")
            )
            if token:
                return str(token)
            env_key = (
                meta.get("env_key")
                or meta.get("envKey")
                or attrs.get("env_key")
                or attrs.get("envKey")
            )
            if env_key:
                return os.getenv(str(env_key).strip())
        return None

    def transport_overrides(self) -> Dict[str, str]:
        """
        Return per-auth transport overrides (e.g., proxy, cert/key paths).
        Reads from attributes first, then metadata. Supports custom transport factory keys.
        """
        overrides: Dict[str, str] = {}
        for key in (
            "proxy",
            "cert",
            "key",
            "mtls_cert",
            "mtls_key",
            "transport_factory",
            "transport_factory_key",
        ):
            if key in self.attributes and self.attributes[key]:
                overrides[key] = self.attributes[key]
            elif key in self.metadata and self.metadata[key]:
                overrides[key] = self.metadata[key]
        # normalise mTLS aliases
        if "mtls_cert" in overrides and "cert" not in overrides:
            overrides["cert"] = overrides["mtls_cert"]
        if "mtls_key" in overrides and "key" not in overrides:
            overrides["key"] = overrides["mtls_key"]
        return overrides


@dataclass
class RequestContext:
    """
    Context passed to auth strategies so they can tailor headers or routing.

    Fields intentionally mirror router inputs and can be extended as needed.
    """

    model: str
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuthStore(Protocol):
    """
    Persistence interface for credentials.

    Namespaces allow multi-tenant separation (e.g., team/user).
    Implementations should be thread-safe and durable.
    """

    def get(self, namespace: str, auth_id: str) -> Optional[AuthRecord]:
        ...

    def save(self, namespace: str, record: AuthRecord) -> AuthRecord:
        ...

    def delete(self, namespace: str, auth_id: str) -> None:
        ...

    def list(self, namespace: str) -> List[AuthRecord]:
        ...


class AuthStrategy(Protocol):
    """
    Provider-specific strategy for preparing and refreshing auth.

    Implementations should be stateless; all runtime state lives in AuthRecord.
    """

    provider: str

    def supports(self, model: str) -> bool:
        """Return True if the strategy can serve this model."""
        ...

    def prepare(
        self, headers: Dict[str, str], ctx: RequestContext, auth: AuthRecord
    ) -> Dict[str, str]:
        """
        Inject auth headers for a request.

        Should not mutate the passed headers; return a new dict with updates.
        """
        ...

    def expiration(self, auth: AuthRecord) -> Optional[datetime]:
        """Return token expiry if known."""
        ...

    def refresh_lead(self, auth: AuthRecord) -> Optional[timedelta]:
        """Optional lead time before expiry to trigger proactive refresh."""
        ...

    def refresh(self, auth: AuthRecord, ctx: RequestContext) -> AuthRecord:
        """Return an updated AuthRecord with refreshed tokens."""
        ...


_EXPIRE_KEYS = ("expired", "expire", "expires_at", "expiresAt", "expiry", "expires")


def expiration_from_metadata(meta: Optional[Dict[str, Any]]) -> Optional[datetime]:
    """
    Parse expiry timestamps from common metadata shapes, including nested token maps.
    """
    if not meta:
        return None

    def parse_value(v: Any) -> Optional[datetime]:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, (int, float)):
            if v <= 0:
                return None
            # heuristic: milliseconds vs seconds
            if v > 1_000_000_000_000:
                return datetime.fromtimestamp(v / 1000, tz=timezone.utc)
            return datetime.fromtimestamp(v, tz=timezone.utc)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            for layout in [
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
            ]:
                try:
                    return datetime.strptime(s, layout)
                except Exception:
                    continue
            try:
                # numeric string fallback
                num = float(s)
                return parse_value(num)
            except Exception:
                return None
        return None

    for key in _EXPIRE_KEYS:
        if key in meta:
            ts = parse_value(meta[key])
            if ts:
                return ts

    for nested_key in ("token", "Token"):
        nested = meta.get(nested_key)
        if isinstance(nested, dict):
            ts = expiration_from_metadata(nested)
            if ts:
                return ts
    return None


def merge_aliases(
    base: Mapping[str, Iterable[str]],
    overlay: Mapping[str, Iterable[str]],
) -> Dict[str, List[str]]:
    """
    Merge two model-alias maps, preserving order and uniqueness.

    Returns a new dict keyed by logical model -> list of provider-specific names.
    """
    merged: Dict[str, List[str]] = {
        logical: list(aliases) for logical, aliases in base.items()
    }
    for logical, aliases in overlay.items():
        bucket = merged.setdefault(logical, [])
        for alias in aliases:
            if alias not in bucket:
                bucket.append(alias)
    return merged


_refresh_lead_factories: Dict[str, Any] = {}


def register_refresh_lead(provider: str, factory) -> None:
    """
    Register provider-specific refresh lead.

    factory may accept (AuthRecord) or no args; should return timedelta or None.
    """
    key = provider.strip().lower()
    if not key or factory is None:
        return
    _refresh_lead_factories[key] = factory


def provider_refresh_lead(provider: str, auth: AuthRecord) -> Optional[timedelta]:
    key = provider.strip().lower()
    factory = _refresh_lead_factories.get(key)
    if factory is None:
        return None
    try:
        return factory(auth) if callable(factory) else None
    except Exception:
        return None


_refresh_evaluator_factories: Dict[str, Any] = {}
_refresh_backoffs: Dict[str, Dict[str, int]] = {}


def register_refresh_evaluator(provider: str, evaluator) -> None:
    """
    Register per-provider evaluator callable (AuthRecord -> bool) to decide refresh.
    """
    key = provider.strip().lower()
    if not key or evaluator is None:
        return
    _refresh_evaluator_factories[key] = evaluator


def refresh_evaluator_for(provider: str):
    return _refresh_evaluator_factories.get(provider.strip().lower())


def register_refresh_backoff(provider: str, pending_seconds: int, failure_seconds: int) -> None:
    key = provider.strip().lower()
    if not key:
        return
    _refresh_backoffs[key] = {
        "pending": max(10, pending_seconds),
        "failure": max(10, failure_seconds),
    }


def refresh_backoff_for(provider: str) -> Optional[Dict[str, int]]:
    return _refresh_backoffs.get(provider.strip().lower())
