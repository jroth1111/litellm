"""
Background maintenance for AuthRecords.

Features:
- Periodic preemptive refresh when expiry is within refresh_lead.
- Periodic health reset: re-enable auth/model when cooldown/backoff has elapsed.

Usage:
    maint = AuthMaintainer(store, namespace="default", strategies={...})
    maint.start()
    ...
    maint.stop()
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from .core import (
    AuthRecord,
    AuthStatus,
    AuthStore,
    AuthStrategy,
    RequestContext,
    provider_refresh_lead,
    refresh_backoff_for,
    refresh_evaluator_for,
)
from .selector import interpret_status_code


class AuthMaintainer:
    def __init__(
        self,
        store: AuthStore,
        namespace: str,
        strategies: Dict[str, AuthStrategy],
        interval_seconds: int = 60,
        refresh_evaluator: Optional[callable] = None,
        refresh_failure_backoff: int = 300,
        refresh_pending_backoff: int = 60,
    ) -> None:
        self.store = store
        self.namespace = namespace
        self.strategies = strategies
        self.interval_seconds = max(10, interval_seconds)
        self.refresh_evaluator = refresh_evaluator
        self.refresh_failure_backoff = max(10, refresh_failure_backoff)
        self.refresh_pending_backoff = max(10, refresh_pending_backoff)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._sweep()
            except Exception:
                # best-effort; swallow to avoid killing the thread
                pass
            self._stop_event.wait(self.interval_seconds)

    def _sweep(self) -> None:
        now = datetime.now(timezone.utc)
        records = self.store.list(self.namespace)
        for record in records:
            strat = self.strategies.get(record.provider)
            if strat is None:
                continue
            updated = self._refresh_if_needed(record, strat, now)
            updated = self._reset_health_if_needed(updated, now)
            if updated is not record:
                self.store.save(self.namespace, updated)

    def _refresh_if_needed(
        self, record: AuthRecord, strat: AuthStrategy, now: datetime
    ) -> AuthRecord:
        exp = strat.expiration(record)
        lead = strat.refresh_lead(record) or provider_refresh_lead(record.provider, record)
        if exp is None or lead is None:
            return record
        if exp - now > lead:
            return record
        evaluator = self.refresh_evaluator
        provider_eval = refresh_evaluator_for(record.provider)
        if evaluator and not evaluator(record, strat, now):
            return record
        if provider_eval and not provider_eval(record, strat, now):
            return record
        if record.next_refresh_after and record.next_refresh_after > now:
            return record
        try:
            refreshed = strat.refresh(
                record,
                ctx=RequestContext(model=""),
            )
            refreshed.status = AuthStatus.ACTIVE
            refreshed.unavailable = False
            refreshed.status_message = ""
            refreshed.last_refreshed_at = now
            refreshed.next_refresh_after = None
            return refreshed
        except Exception:
            backoff_cfg = refresh_backoff_for(record.provider)
            failure_backoff = (
                backoff_cfg.get("failure", self.refresh_failure_backoff)
                if backoff_cfg
                else self.refresh_failure_backoff
            )
            updated = record.clone()
            updated.next_refresh_after = now + timedelta(seconds=failure_backoff)
            return updated

    def _reset_health_if_needed(
        self, record: AuthRecord, now: datetime
    ) -> AuthRecord:
        changed = False
        updated = record
        if record.unavailable and record.next_retry_after and record.next_retry_after <= now:
            updated = record.clone()
            updated.unavailable = False
            updated.status = AuthStatus.ACTIVE
            updated.status_message = ""
            updated.next_retry_after = None
            changed = True
        for model_key, state in list(updated.model_states.items()):
            if (
                state.unavailable
                and state.next_retry_after
                and state.next_retry_after <= now
            ):
                if not changed:
                    updated = updated.clone()
                    changed = True
                ms = updated.model_states[model_key]
                ms.unavailable = False
                ms.status = AuthStatus.ACTIVE
                ms.status_message = ""
                ms.next_retry_after = None
                ms.quota.exceeded = False
        return updated
