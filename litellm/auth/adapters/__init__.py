"""
Subscription OAuth adapter layer.

Adapters consolidate provider-specific subscription logic:
- CLI login flows (browser/device/poll)
- AuthStrategy behavior for Router (prepare/refresh)
- Optional capabilities like model discovery
"""

from .base import AdapterCapabilities, LoginFlow
from .registry import get_adapter, list_adapters

__all__ = [
    "AdapterCapabilities",
    "LoginFlow",
    "get_adapter",
    "list_adapters",
]

