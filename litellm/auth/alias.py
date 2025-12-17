"""
Model alias mapping utilities.

Keeps logical model names separate from provider-specific model identifiers,
allowing failover across providers that serve equivalent models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping


@dataclass
class ModelAliasMap:
    """
    Maps logical model names to ordered provider-specific model identifiers.

    Example:
        aliases = ModelAliasMap()
        aliases.register("claude-3.5-sonnet", "anthropic/claude-3.5-sonnet", "openrouter/anthropic/claude-3.5-sonnet")
    """

    aliases: Dict[str, List[str]] = field(default_factory=dict)

    def register(self, logical: str, *provider_models: str) -> None:
        """Register one or more provider-specific names for a logical model."""
        bucket = self.aliases.setdefault(logical, [])
        for name in provider_models:
            if name not in bucket:
                bucket.append(name)

    def resolve(self, logical: str) -> List[str]:
        """Return ordered provider model names for a logical model."""
        return list(self.aliases.get(logical, [logical]))

    def merge(self, mapping: Mapping[str, Iterable[str]]) -> None:
        """Merge in an external mapping, preserving order and uniqueness."""
        for logical, provider_models in mapping.items():
            self.register(logical, *provider_models)

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Iterable[str]]) -> "ModelAliasMap":
        inst = cls()
        inst.merge(mapping)
        return inst
