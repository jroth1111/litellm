"""
Subscription-specific metrics helpers.

Provides counters for auth usage and rate-limit violations. Supports
optional Prometheus export when prometheus_client is installed.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, DefaultDict, Optional, TYPE_CHECKING

from .core import AuthRecord

__all__ = [
    "AuthMetrics",
    "AuthHooks",
    "NoopAuthHooks",
    "PrometheusAuthMetrics",
]

if TYPE_CHECKING:
    from prometheus_client import Counter


class AuthMetrics:
    """In-memory auth metrics with optional sink callback."""

    def __init__(self, sink: Callable[..., Any] | None = None) -> None:
        self.calls_by_provider: DefaultDict[str, int] = defaultdict(int)
        self.quota_hits_by_provider: DefaultDict[str, int] = defaultdict(int)
        self.auth_errors_by_provider: DefaultDict[str, int] = defaultdict(int)
        self.sink = sink  # optional hook to emit custom metrics

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


class PrometheusAuthMetrics(AuthMetrics):
    """
    AuthMetrics that exports to Prometheus counters.

    Requires prometheus_client to be installed. If not available, falls back
    to in-memory counting only.

    Counters exposed:
    - litellm_auth_calls_total{provider}
    - litellm_auth_quota_hits_total{provider}
    - litellm_auth_errors_total{provider}
    """

    def __init__(self) -> None:
        super().__init__()
        self._calls_counter: Optional["Counter"] = None
        self._quota_counter: Optional["Counter"] = None
        self._errors_counter: Optional["Counter"] = None

        try:
            from prometheus_client import Counter

            self._calls_counter = Counter(
                "litellm_auth_calls_total",
                "Total successful auth calls by provider",
                ["provider"],
            )
            self._quota_counter = Counter(
                "litellm_auth_quota_hits_total",
                "Auth quota exhaustion events by provider",
                ["provider"],
            )
            self._errors_counter = Counter(
                "litellm_auth_errors_total",
                "Auth errors by provider",
                ["provider"],
            )
        except ImportError:
            pass  # prometheus_client not installed; use in-memory only

    def record_call(self, provider: str) -> None:
        super().record_call(provider)
        if self._calls_counter is not None:
            self._calls_counter.labels(provider=provider).inc()

    def record_quota_hit(self, provider: str) -> None:
        super().record_quota_hit(provider)
        if self._quota_counter is not None:
            self._quota_counter.labels(provider=provider).inc()

    def record_auth_error(self, provider: str) -> None:
        super().record_auth_error(provider)
        if self._errors_counter is not None:
            self._errors_counter.labels(provider=provider).inc()


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

