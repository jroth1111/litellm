from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

from .anthropic import AnthropicSubscriptionAdapter
from .antigravity import AntigravitySubscriptionAdapter
from .cursor import CursorSubscriptionAdapter
from .gemini import GeminiSubscriptionAdapter
from .github_copilot import GitHubCopilotSubscriptionAdapter
from .openai_chatgpt import OpenAIChatGPTSubscriptionAdapter
from .qwen import QwenSubscriptionAdapter


_logger = logging.getLogger(__name__)
_EXTRA_ADAPTER_PATHS: List[str] = []


def register_adapter_paths(paths: Iterable[str]) -> None:
    """Register additional adapter import paths (module:attr)."""
    global _EXTRA_ADAPTER_PATHS
    _EXTRA_ADAPTER_PATHS = [p.strip() for p in paths if isinstance(p, str) and p.strip()]


def reset_adapter_paths() -> None:
    """Clear registered adapter import paths (test-only)."""
    global _EXTRA_ADAPTER_PATHS
    _EXTRA_ADAPTER_PATHS = []


def _extra_adapter_paths() -> List[str]:
    paths = list(_EXTRA_ADAPTER_PATHS)
    env = os.getenv("LITELLM_AUTH_ADAPTERS", "")
    if env:
        paths.extend([p.strip() for p in env.split(",") if p.strip()])
    # preserve order, de-dupe
    seen = set()
    deduped: List[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _load_adapter_from_path(path: str) -> Optional[object]:
    module_name, attr = (path.split(":", 1) + [None])[:2]
    if not module_name:
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        _logger.debug("auth adapter import failed for %s: %s", path, exc)
        return None
    target = None
    if attr:
        target = getattr(module, attr, None)
    else:
        for candidate in ("ADAPTER", "adapter", "Adapter"):
            target = getattr(module, candidate, None)
            if target is not None:
                break
    if target is None:
        return None
    if isinstance(target, type):
        try:
            return target()
        except Exception as exc:
            _logger.debug("auth adapter init failed for %s: %s", path, exc)
            return None
    if callable(target) and not hasattr(target, "provider"):
        try:
            return target()
        except Exception as exc:
            _logger.debug("auth adapter factory failed for %s: %s", path, exc)
            return None
    return target


def list_adapters() -> List[object]:
    adapters: List[object] = [
        AnthropicSubscriptionAdapter(),
        OpenAIChatGPTSubscriptionAdapter(),
        GeminiSubscriptionAdapter(),
        GitHubCopilotSubscriptionAdapter(),
        AntigravitySubscriptionAdapter(),
        QwenSubscriptionAdapter(),
        CursorSubscriptionAdapter(),
    ]
    extras: List[object] = []
    for path in _extra_adapter_paths():
        adapter = _load_adapter_from_path(path)
        if adapter is not None:
            extras.append(adapter)
    if not extras:
        return adapters
    merged: Dict[str, object] = {}
    for adapter in adapters + extras:
        provider = getattr(adapter, "provider", None)
        if provider:
            merged[str(provider).strip().lower()] = adapter
    return list(merged.values())


def get_adapter(provider: str) -> Optional[object]:
    key = (provider or "").strip().lower()
    for adapter in list_adapters():
        if getattr(adapter, "provider", "").lower() == key:
            return adapter
    return None


def default_strategies() -> Dict[str, object]:
    """
    Return the default provider->AuthStrategy mapping used by the proxy/router.

    Adapters implement the AuthStrategy protocol so they can be plugged in
    directly without wrapper strategy classes.
    """
    return {getattr(a, "provider"): a for a in list_adapters()}


def list_provider_descriptors() -> List[Dict[str, Any]]:
    """
    Public, stable provider list for CLI + management APIs.

    This replaces the legacy provider_registry module and is derived from
    adapters (single source of truth).
    """
    out: List[Dict[str, Any]] = []
    for adapter in list_adapters():
        provider = getattr(adapter, "provider", None)
        if not provider:
            continue
        caps = getattr(adapter, "capabilities", None)
        out.append(
            {
                "provider": provider,
                "login_flow": getattr(caps, "login_flow", None),
                "supports_refresh": bool(
                    getattr(caps, "supports_refresh", getattr(adapter, "supports_refresh", True))
                ),
                "supports_models_list": bool(getattr(caps, "supports_models_list", False)),
                "methods": getattr(adapter, "methods", lambda: [])(),
            }
        )
    out.sort(key=lambda item: str(item.get("provider", "")).lower())
    return out
