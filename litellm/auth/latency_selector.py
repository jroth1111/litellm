"""
Latency-aware selector example that prefers auths with lower observed latency.
Requires auth.metadata["latency_ms"] to be optionally populated externally.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple

from .alias import ModelAliasMap
from .core import AuthRecord
from .selector import AuthSelectionResult, CredentialSelector


class LatencySelector(CredentialSelector):
    def __init__(
        self,
        alias_map: Optional[ModelAliasMap] = None,
        disable_quota_cooldown: bool = False,
    ) -> None:
        super().__init__(alias_map=alias_map, disable_quota_cooldown=disable_quota_cooldown)

    def _prioritize(
        self,
        logical_model: str,
        candidates: Iterable[Tuple[AuthRecord, str]],
        now: datetime,
        preferred_providers: Optional[List[str]] = None,
    ) -> Tuple[AuthRecord, str]:
        base_selected = super()._prioritize(
            logical_model, candidates, now, preferred_providers
        )
        # Prefer lower latency within same provider/model
        best = base_selected
        best_latency = self._latency(base_selected[0])
        for cand in candidates:
            if cand[0].provider != base_selected[0].provider:
                continue
            lat = self._latency(cand[0])
            if lat is not None and (best_latency is None or lat < best_latency):
                best_latency = lat
                best = cand
        return best

    @staticmethod
    def _latency(auth: AuthRecord) -> Optional[float]:
        meta = auth.metadata or {}
        try:
            lat = meta.get("latency_ms")
            if lat is None:
                return None
            return float(lat)
        except Exception:
            return None
