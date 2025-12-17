"""
Weighted selector example that prefers higher weight auths per provider.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple

from .alias import ModelAliasMap
from .core import AuthRecord
from .selector import AuthSelectionResult, CredentialSelector, Selector


class WeightedSelector(CredentialSelector):
    def __init__(
        self,
        alias_map: Optional[ModelAliasMap] = None,
        weights: Optional[dict[str, int]] = None,
        disable_quota_cooldown: bool = False,
    ) -> None:
        super().__init__(alias_map=alias_map, disable_quota_cooldown=disable_quota_cooldown)
        self.weights = weights or {}

    def _prioritize(
        self,
        logical_model: str,
        candidates: Iterable[Tuple[AuthRecord, str]],
        now: datetime,
        preferred_providers: Optional[List[str]] = None,
    ) -> Tuple[AuthRecord, str]:
        # Apply base prioritization then weights to break ties
        base_selected = super()._prioritize(
            logical_model, candidates, now, preferred_providers
        )
        # If multiple from same provider, prefer higher weight
        same_provider = [
            c for c in candidates if c[0].provider == base_selected[0].provider
        ]
        if len(same_provider) <= 1:
            return base_selected
        best = base_selected
        best_weight = self.weights.get(base_selected[0].id, 0)
        for cand in same_provider:
            w = self.weights.get(cand[0].id, 0)
            if w > best_weight:
                best_weight = w
                best = cand
        return best
