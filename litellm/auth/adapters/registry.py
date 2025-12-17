from __future__ import annotations

from typing import Any, Dict, List, Optional

from .anthropic import AnthropicSubscriptionAdapter
from .antigravity import AntigravitySubscriptionAdapter
from .cursor import CursorSubscriptionAdapter
from .gemini import GeminiSubscriptionAdapter
from .github_copilot import GitHubCopilotSubscriptionAdapter
from .openai_chatgpt import OpenAIChatGPTSubscriptionAdapter
from .qwen import QwenSubscriptionAdapter


def list_adapters() -> List[object]:
    return [
        AnthropicSubscriptionAdapter(),
        OpenAIChatGPTSubscriptionAdapter(),
        GeminiSubscriptionAdapter(),
        GitHubCopilotSubscriptionAdapter(),
        AntigravitySubscriptionAdapter(),
        QwenSubscriptionAdapter(),
        CursorSubscriptionAdapter(),
    ]


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
            }
        )
    out.sort(key=lambda item: str(item.get("provider", "")).lower())
    return out
