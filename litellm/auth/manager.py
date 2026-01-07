"""
Lightweight auth manager orchestrating selection, refresh, and hooks.

This mirrors the CLIProxyAPIPlus pattern at a smaller scale:
- Owns selector + strategies + hooks
- Supports refresh with backoff and runtime preservation
- Optional background maintainer for preemptive refresh and cooldown reset
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

from .core import AuthRecord, AuthStatus, AuthStore, AuthStrategy, provider_refresh_lead
from .maintenance import AuthMaintainer
from .metrics import AuthHooks, NoopAuthHooks
from .selector import CredentialSelector, retry_after_from_exception


class AuthManager:
    def __init__(
        self,
        store: AuthStore,
        strategies: Dict[str, AuthStrategy],
        selector: Optional[CredentialSelector] = None,
        hooks: Optional[AuthHooks] = None,
        namespace: str = "default",
        maintainer_interval_seconds: int = 60,
        refresh_evaluators: Optional[Dict[str, callable]] = None,
        refresh_backoffs: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> None:
        self.store = store
        self.strategies = strategies
        self.selector = selector or CredentialSelector()
        self.hooks = hooks or NoopAuthHooks()
        self.namespace = namespace
        self.maintainer_interval = maintainer_interval_seconds
        self.refresh_evaluators = refresh_evaluators or {}
        self.refresh_backoffs = refresh_backoffs or {}
        self._maintainer: Optional[AuthMaintainer] = None
        self._refresh_inflight: set[str] = set()
        self._lock = threading.Lock()

    def start_auto_refresh(self) -> None:
        if self._maintainer is None:
            self._maintainer = AuthMaintainer(
                store=self.store,
                namespace=self.namespace,
                strategies=self.strategies,
                interval_seconds=self.maintainer_interval,
                refresh_evaluator=self._merged_refresh_evaluator(),
                refresh_failure_backoff=self._merged_backoff("failure"),
                refresh_pending_backoff=self._merged_backoff("pending"),
            )
        self._maintainer.start()

    def stop_auto_refresh(self) -> None:
        if self._maintainer:
            self._maintainer.stop()

    def register(self, auth: AuthRecord) -> AuthRecord:
        saved = self.store.save(self.namespace, auth)
        try:
            self.hooks.on_register(saved)
        except Exception:
            pass
        return saved

    def update(self, auth: AuthRecord, reason: str = "") -> AuthRecord:
        saved = self.store.save(self.namespace, auth)
        try:
            self.hooks.on_update(saved, reason=reason)
        except Exception:
            pass
        return saved

    def list(self) -> List[AuthRecord]:
        return self.store.list(self.namespace)

    def get(self, auth_id: str) -> Optional[AuthRecord]:
        return self.store.get(self.namespace, auth_id)

    def select(
        self,
        logical_model: str,
        auth_records: Optional[Sequence[AuthRecord]] = None,
        allow_cross_provider: bool = True,
        preferred_providers: Optional[List[str]] = None,
    ):
        records = auth_records or self.list()
        return self.selector.select(
            logical_model=logical_model,
            auth_records=records,
            allow_cross_provider=allow_cross_provider,
            preferred_providers=preferred_providers,
        )

    def refresh_auth(self, auth: AuthRecord, strategy: AuthStrategy) -> AuthRecord:
        refreshed = strategy.refresh(auth, ctx=None)  # type: ignore[arg-type]
        # Preserve runtime from prior record
        refreshed.runtime = auth.runtime
        refreshed.updated_at = datetime.now(timezone.utc)
        refreshed.last_refreshed_at = datetime.now(timezone.utc)
        saved = self.store.save(self.namespace, refreshed)
        try:
            self.hooks.on_update(saved, reason="refresh")
        except Exception:
            pass
        return saved

    def refresh_if_due(self, auth: AuthRecord) -> AuthRecord:
        strat = self.strategies.get(auth.provider)
        if strat is None:
            return auth
        exp = strat.expiration(auth)
        lead = strat.refresh_lead(auth) or provider_refresh_lead(auth.provider, auth)
        now = datetime.now(timezone.utc)
        if exp is None or lead is None:
            return auth
        if exp - now > lead:
            return auth
        provider_eval = self.refresh_evaluators.get(auth.provider)
        if provider_eval is not None:
            try:
                if not provider_eval(auth, strat, now):
                    return auth
            except Exception:
                return auth
        with self._lock:
            if auth.id in self._refresh_inflight:
                return auth
            self._refresh_inflight.add(auth.id)
        try:
            updated = self.refresh_auth(auth, strat)
            return updated
        except Exception as e:
            # Graceful degradation: keep token usable if not yet expired
            # Only record the error message and schedule retry
            failed = auth.clone()
            backoff = self._merged_backoff("failure")
            failed.next_refresh_after = now + timedelta(seconds=backoff)
            
            # Store last error in status_message for observability
            error_msg = str(e) if str(e) else type(e).__name__
            failed.status_message = f"Refresh failed: {error_msg[:200]}"
            
            # Check if token has actually expired
            exp = strat.expiration(auth)
            if exp is not None and exp <= now:
                # Token is expired, mark as error
                failed.status = AuthStatus.EXPIRED
            # else: keep existing status (likely ACTIVE) - token still usable
            
            self.store.save(self.namespace, failed)
            return failed
        finally:
            with self._lock:
                self._refresh_inflight.discard(auth.id)

    def mark_result(
        self,
        auth: AuthRecord,
        provider_model: str,
        success: bool,
        error: Optional[BaseException] = None,
    ) -> AuthRecord:
        if success:
            updated = self.selector.mark_success(
                auth, provider_model, store=self.store, namespace=self.namespace
            )
            try:
                self.hooks.on_success(updated, provider_model, None, False)
            except Exception:
                pass
            return updated
        retry_after = retry_after_from_exception(error) if error else None
        status_code = None
        if error is not None:
            status_code = getattr(error, "status_code", None)
        updated = self.selector.mark_failure(
            auth,
            provider_model,
            error_message=str(error) if error else "unknown error",
            status_code=status_code,
            is_quota=status_code == 429 if status_code else False,
            retry_after=retry_after,
            store=self.store,
            namespace=self.namespace,
        )
        try:
            self.hooks.on_failure(
                updated,
                provider_model,
                error,
                retry_after.total_seconds() if retry_after else None,
                status_code == 429 if status_code else False,
                status_code,
            )
        except Exception:
            pass
        return updated

    def _merged_refresh_evaluator(self):
        def evaluator(record, strat, now):
            provider_eval = self.refresh_evaluators.get(record.provider)
            if provider_eval is None:
                return True
            try:
                return provider_eval(record, strat, now)
            except Exception:
                return True

        return evaluator

    def _merged_backoff(self, kind: str) -> int:
        # choose provider-specific if all entries agree; otherwise default
        if not self.refresh_backoffs:
            return 300 if kind == "failure" else 60
        # fall back to defaults when missing
        if kind == "failure":
            return list(self.refresh_backoffs.values())[0].get("failure", 300)
        return list(self.refresh_backoffs.values())[0].get("pending", 60)
