"""
Selector registry to allow swapping selection strategies without changing call sites.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from .selector import Selector

_selector_registry: Dict[str, Callable[[], Selector]] = {}


def register_selector(name: str, factory: Callable[[], Selector]) -> None:
    key = name.strip().lower()
    if not key or factory is None:
        return
    _selector_registry[key] = factory


def get_selector(name: str) -> Optional[Selector]:
    key = name.strip().lower()
    factory = _selector_registry.get(key)
    if factory is None:
        return None
    try:
        return factory()
    except Exception:
        return None
