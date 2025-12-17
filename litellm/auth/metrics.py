"""
Subscription-specific metrics helpers (stub hooks).

Provides minimal counters for auth usage and rate-limit violations. In a real
deployment, wire these to Prometheus or your metrics system.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, DefaultDict, Optional

from .core import AuthRecord


class AuthMetrics:
    def __init__(self, sink: Callable[..., Any] | None = None) -> None:
        self.calls_by_provider: DefaultDict[str, int] = defaultdict(int)
        self.quota_hits_by_provider: DefaultDict[str, int] = defaultdict(int)
        self.auth_errors_by_provider: DefaultDict[str, int] = defaultdict(int)
        self.sink = sink  # optional hook to emit Prometheus-style metrics

    def record_call(self, provider: str) -> None:
        self.calls_by_provider[provider] += 1
        if self.sink:
            self.sink("auth_calls_total", provider=provider)

    def record_quota_hit(self, provider: str) -> None:
        self.quota_hits_by_provider[provider] += 1
        if self.sink:
            self.sink("auth_quota_hits_total", provider=provider)

    def record_auth_error(self, provider: str) -> None:
        self.auth_errors_by_provider[provider] += 1
        if self.sink:
            self.sink("auth_errors_total", provider=provider)


class AuthHooks:
    """
    Hook interface for auth lifecycle/result events.
    Override methods to emit logs/metrics to your system of choice.
    """

    def on_register(self, auth: AuthRecord) -> None:
        ...

    def on_update(self, auth: AuthRecord, reason: str = "") -> None:
        ...

    def on_success(
        self,
        auth: AuthRecord,
        provider_model: Optional[str] = None,
        retry_after: Optional[float] = None,
        is_quota: bool = False,
    ) -> None:
        ...

    def on_failure(
        self,
        auth: AuthRecord,
        provider_model: Optional[str] = None,
        error: Optional[BaseException] = None,
        retry_after: Optional[float] = None,
        is_quota: bool = False,
        status_code: Optional[int] = None,
    ) -> None:
        ...


class NoopAuthHooks(AuthHooks):
    def on_register(self, auth: AuthRecord) -> None:  # pragma: no cover - trivial
        return

    def on_update(self, auth: AuthRecord, reason: str = "") -> None:  # pragma: no cover
        return

    def on_success(
        self,
        auth: AuthRecord,
        provider_model: Optional[str] = None,
        retry_after: Optional[float] = None,
        is_quota: bool = False,
    ) -> None:  # pragma: no cover
        return

    def on_failure(
        self,
        auth: AuthRecord,
        provider_model: Optional[str] = None,
        error: Optional[BaseException] = None,
        retry_after: Optional[float] = None,
        is_quota: bool = False,
        status_code: Optional[int] = None,
    ) -> None:  # pragma: no cover
        return
