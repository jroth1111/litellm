"""
Credential selector and health management for subscription OAuth flows.

This module is provider-agnostic and operates on AuthRecord data. It
prioritizes:
1) Healthy credentials for the target provider/model.
2) Other accounts on the same provider.
3) Aliased models across providers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, List, Optional, Protocol, Sequence, Tuple

from .alias import ModelAliasMap
from .core import (
    AuthRecord,
    AuthStatus,
    AuthStore,
    ModelState,
    QuotaState,
    RequestContext,
)


@dataclass
class AuthSelectionResult:
    auth: AuthRecord
    provider_model: str


class Selector(Protocol):
    """
    Selector protocol for choosing an AuthRecord for a logical model.
    """

    def select(
        self,
        logical_model: str,
        auth_records: Sequence[AuthRecord],
        now: Optional[datetime] = None,
        allow_cross_provider: bool = True,
        preferred_providers: Optional[List[str]] = None,
    ) -> Optional[AuthSelectionResult]:
        ...


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _model_provider(model: str) -> Optional[str]:
    """
    Extract provider prefix if model uses a namespaced form like "provider/model".
    """
    if "/" not in model:
        return None
    return model.split("/", 1)[0]


def _is_model_healthy(state: Optional[ModelState], now: datetime) -> bool:
    if state is None:
        return True
    if state.status == AuthStatus.DISABLED or state.status == AuthStatus.EXPIRED:
        return False
    if state.unavailable and state.next_retry_after and state.next_retry_after > now:
        return False
    return True


def _is_auth_healthy(auth: AuthRecord, model: str, now: datetime) -> bool:
    # Treat ERROR as a transient state gated by `unavailable/next_retry_after`.
    # Otherwise a single 429 permanently bricks an auth record unless a background
    # maintainer thread resets it.
    if auth.status in {AuthStatus.DISABLED, AuthStatus.EXPIRED}:
        return False
    if auth.unavailable and auth.next_retry_after and auth.next_retry_after > now:
        return False
    model_state = auth.model_states.get(model)
    return _is_model_healthy(model_state, now)


class CredentialSelector:
    def __init__(
        self,
        alias_map: Optional[ModelAliasMap] = None,
        disable_quota_cooldown: bool = False,
        provider_quota_cooldown_overrides: Optional[dict[str, bool]] = None,
        model_quota_cooldown_overrides: Optional[dict[str, bool]] = None,
        offset_store_path: Optional[str] = None,
    ) -> None:
        self.alias_map = alias_map or ModelAliasMap()
        self.offset_store_path = offset_store_path
        self.provider_offsets: dict[str, int] = self._load_offsets()
        self.disable_quota_cooldown = disable_quota_cooldown
        self.provider_quota_cooldown_overrides = provider_quota_cooldown_overrides or {}
        self.model_quota_cooldown_overrides = model_quota_cooldown_overrides or {}

    def _load_offsets(self) -> dict[str, int]:
        """Load rotation offsets from file if configured."""
        if not self.offset_store_path:
            return {}
        try:
            if os.path.exists(self.offset_store_path):
                with open(self.offset_store_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return {k: int(v) for k, v in data.items() if isinstance(v, int)}
        except Exception:
            pass
        return {}

    def _save_offsets(self) -> None:
        """Persist rotation offsets to file if configured."""
        if not self.offset_store_path:
            return
        try:
            # Ensure parent directory exists
            parent = os.path.dirname(self.offset_store_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            with open(self.offset_store_path, "w") as f:
                json.dump(self.provider_offsets, f)
        except Exception:
            pass  # Best-effort persistence; don't fail requests

    def select(
        self,
        logical_model: str,
        auth_records: Sequence[AuthRecord],
        now: Optional[datetime] = None,
        allow_cross_provider: bool = True,
        preferred_providers: Optional[List[str]] = None,
    ) -> Optional[AuthSelectionResult]:
        """
        Choose a credential for the target logical model.

        Preference order:
        - healthy credential matching target provider for each alias
        - other accounts of same provider
        - healthy credentials for aliased models across providers
        """
        if not auth_records:
            return None
        ts = now or _now()
        aliases = self.alias_map.resolve(logical_model)
        primary_provider = _model_provider(aliases[0]) if aliases else None
        if not allow_cross_provider and primary_provider:
            aliases = [a for a in aliases if _model_provider(a) == primary_provider]
        # 1) strict provider match pass
        for alias in aliases:
            provider = _model_provider(alias)
            candidates: List[Tuple[AuthRecord, str]] = []
            for auth in auth_records:
                if provider and auth.provider != provider:
                    continue
                if _is_auth_healthy(auth, alias, ts):
                    candidates.append((auth, alias))
            if candidates:
                auth, provider_model = self._prioritize(
                    logical_model, candidates, ts, preferred_providers
                )
                return AuthSelectionResult(auth=auth, provider_model=provider_model)
        # 2) relaxed pass across providers
        if allow_cross_provider:
            relaxed_candidates: List[Tuple[AuthRecord, str]] = []
            for alias in aliases:
                for auth in auth_records:
                    if _is_auth_healthy(auth, alias, ts):
                        relaxed_candidates.append((auth, alias))
            if relaxed_candidates:
                auth, provider_model = self._prioritize(
                    logical_model, relaxed_candidates, ts, preferred_providers
                )
                return AuthSelectionResult(auth=auth, provider_model=provider_model)
        return None

    def _prioritize(
        self,
        logical_model: str,
        candidates: Iterable[Tuple[AuthRecord, str]],
        now: datetime,
        preferred_providers: Optional[List[str]] = None,
    ) -> Tuple[AuthRecord, str]:
        """
        Pick the best candidate based on preferred providers, soonest retry, and most recent refresh.
        Rotate within the same provider to spread load across accounts.
        """
        preferred = preferred_providers or []

        def sort_key(item: Tuple[AuthRecord, str]):
            auth, model = item
            model_state = auth.model_states.get(model)
            next_retry = model_state.next_retry_after if model_state else auth.next_retry_after
            if next_retry is None:
                next_retry = datetime.min.replace(tzinfo=timezone.utc)
            refreshed = auth.last_refreshed_at or datetime.min.replace(
                tzinfo=timezone.utc
            )
            preferred_rank = preferred.index(auth.provider) if auth.provider in preferred else len(preferred)
            return (preferred_rank, next_retry, -int(refreshed.timestamp()))

        # group by provider
        provider_buckets: dict[str, List[Tuple[AuthRecord, str]]] = {}
        for item in candidates:
            provider_buckets.setdefault(item[0].provider, []).append(item)
        # sort within provider
        for provider, bucket in provider_buckets.items():
            provider_buckets[provider] = sorted(bucket, key=sort_key)

        # provider order respecting preferred
        provider_order: List[str] = []
        seen = set()
        for p in preferred:
            if p in provider_buckets and p not in seen:
                provider_order.append(p)
                seen.add(p)
        for p in provider_buckets:
            if p not in seen:
                provider_order.append(p)

        rotated: List[Tuple[AuthRecord, str]] = []
        for provider in provider_order:
            bucket = provider_buckets.get(provider, [])
            if not bucket:
                continue
            key = f"{logical_model}:{provider}"
            offset = self.provider_offsets.get(key, 0)
            if len(bucket) > 1:
                offset = offset % len(bucket)
                rotated_bucket = bucket[offset:] + bucket[:offset]
                self.provider_offsets[key] = (offset + 1) % len(bucket)
                rotated.extend(rotated_bucket)
            else:
                rotated.extend(bucket)

        if not rotated:
            rotated = sorted(candidates, key=sort_key)
        # Persist offset updates (best-effort)
        self._save_offsets()
        return rotated[0]

    def mark_success(
        self,
        auth: AuthRecord,
        provider_model: str,
        store: Optional[AuthStore] = None,
        namespace: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> AuthRecord:
        ts = now or _now()
        updated = auth.clone()
        updated.request_count = int(updated.request_count) + 1
        updated.last_request_at = ts
        updated.status = AuthStatus.ACTIVE
        updated.unavailable = False
        updated.next_retry_after = None
        updated.status_message = ""
        updated.quota = QuotaState(exceeded=False)
        state = updated.model_states.get(provider_model) or ModelState()
        state.status = AuthStatus.ACTIVE
        state.unavailable = False
        state.next_retry_after = None
        state.last_error = None
        state.updated_at = ts
        updated.model_states[provider_model] = state
        updated.updated_at = ts
        if store and namespace:
            store.save(namespace, updated)
        return updated

    def mark_failure(
        self,
        auth: AuthRecord,
        provider_model: str,
        error_message: str,
        status_code: Optional[int] = None,
        is_quota: bool = False,
        retry_after: Optional[timedelta] = None,
        store: Optional[AuthStore] = None,
        namespace: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> AuthRecord:
        ts = now or _now()
        updated = auth.clone()
        updated.request_count = int(updated.request_count) + 1
        updated.error_count = int(updated.error_count) + 1
        updated.last_request_at = ts
        # Optional: skip quota cooldowns entirely (useful in tests)
        disable_cooldown = self.disable_quota_cooldown
        if auth.provider in self.provider_quota_cooldown_overrides:
            disable_cooldown = self.provider_quota_cooldown_overrides[auth.provider]
        if provider_model in self.model_quota_cooldown_overrides:
            disable_cooldown = self.model_quota_cooldown_overrides[provider_model]
        if is_quota and disable_cooldown:
            updated.updated_at = ts
            updated.status_message = error_message
            state = updated.model_states.get(provider_model) or ModelState()
            state.status_message = error_message
            state.last_error = error_message
            updated.model_states[provider_model] = state
            if store and namespace:
                store.save(namespace, updated)
            return updated

        updated.status = AuthStatus.ERROR
        updated.status_message = error_message
        updated.unavailable = True
        backoff = _next_backoff(updated.quota.backoff_level, is_quota)
        if retry_after and retry_after > backoff:
            backoff = retry_after
        updated.quota = QuotaState(
            exceeded=is_quota,
            reason=error_message,
            next_recover_at=ts + backoff,
            backoff_level=min(updated.quota.backoff_level + 1, 6),
        )
        updated.next_retry_after = ts + backoff
        state = updated.model_states.get(provider_model) or ModelState()
        state.status = AuthStatus.ERROR
        state.status_message = error_message
        state.unavailable = True
        state.last_error = error_message
        state.next_retry_after = ts + backoff
        state.quota = QuotaState(
            exceeded=is_quota,
            reason=error_message,
            next_recover_at=ts + backoff,
            backoff_level=min(state.quota.backoff_level + 1, 6),
        )
        state.updated_at = ts
        updated.model_states[provider_model] = state
        updated.updated_at = ts
        # Treat 401/403 as refresh-needed; strategy layer can decide to refresh on next use.
        if status_code in (401, 403, 429):
            updated.next_refresh_after = ts
        if store and namespace:
            store.save(namespace, updated)
        return updated


def _next_backoff(level: int, is_quota: bool) -> timedelta:
    """
    Exponential backoff with cap; start higher for quota errors.
    """
    base = 30 if not is_quota else 60
    cap_seconds = 15 * 60
    delay = base * (2 ** level)
    if delay > cap_seconds:
        delay = cap_seconds
    return timedelta(seconds=delay)


def interpret_status_code(exc: BaseException) -> Optional[int]:
    """
    Best-effort extraction of HTTP status code from common exception shapes.
    """
    for attr in ("status_code", "http_status", "status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    # httpx-like responses on exceptions
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _parse_retry_after(value: Any) -> Optional[timedelta]:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            if value <= 0:
                return None
            return timedelta(seconds=float(value))
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            # numeric seconds
            if s.isdigit():
                return timedelta(seconds=float(s))
            from email.utils import parsedate_to_datetime

            try:
                dt = parsedate_to_datetime(s)
                if dt:
                    return dt - datetime.now(timezone.utc)
            except Exception:
                return None
    except Exception:
        return None
    return None


def retry_after_from_exception(exc: BaseException) -> Optional[timedelta]:
    """
    Extract Retry-After hints from common exception/response shapes.
    """
    for attr in ("retry_after", "retry_after_ms"):
        val = getattr(exc, attr, None)
        if val is not None:
            return _parse_retry_after(val if attr == "retry_after" else float(val) / 1000)
    response = getattr(exc, "response", None)
    headers = None
    if response is not None:
        headers = getattr(response, "headers", None)
    if headers and isinstance(headers, dict):
        for key in headers:
            if str(key).lower() == "retry-after":
                return _parse_retry_after(headers[key])
    # some exceptions carry headers attribute directly
    headers = getattr(exc, "headers", None)
    if headers and isinstance(headers, dict):
        for key in headers:
            if str(key).lower() == "retry-after":
                return _parse_retry_after(headers[key])
    return None
