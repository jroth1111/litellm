"""
Core authentication abstractions for subscription and consumer OAuth flows.

This module only defines data structures and interfaces; concrete strategies
and stores live in provider-specific modules.
"""

from .core import (
    AuthRecord,
    AuthStatus,
    AuthStore,
    AuthStrategy,
    ModelState,
    RequestContext,
    QuotaState,
)
from .file_store import EncryptedJsonFileAuthStore, JsonFileAuthStore
from .paths import default_auth_store_dir, default_token_store_dir
from .alias import ModelAliasMap
from .mirror_store import MirrorAuthStore
from .maintenance import AuthMaintainer
from .memory_store import InMemoryAuthStore
try:
    from .redis_store import RedisAuthStore
except Exception:  # pragma: no cover
    class RedisAuthStore:  # type: ignore[dead-code]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "RedisAuthStore requires the `redis` package. Install with `pip install redis` "
                "or avoid importing/using RedisAuthStore."
            )
from .git_store import GitAuthStore
from .object_store import ObjectAuthStore
from .manager import AuthManager
from .metrics import AuthHooks, AuthMetrics, NoopAuthHooks
from .token_storage import (
    EncryptedJsonTokenStorage,
    InMemoryTokenStorage,
    JsonTokenStorage,
    PlaintextJsonTokenStorage,
    TokenRecord,
    TokenStorage,
)
from .weighted_selector import WeightedSelector
from .latency_selector import LatencySelector
from .selector_registry import register_selector, get_selector
from .device_flow import (
    start_device_flow,
    poll_device_flow,
    mark_device_flow_authorized,
)
from .token_management import export_tokens, import_tokens
from .management import (
    delete_auth,
    export_auths,
    get_auth,
    import_auths,
    list_auths,
    save_auth,
)

__all__ = [
    "AuthRecord",
    "AuthStatus",
    "AuthStore",
    "AuthStrategy",
    "JsonFileAuthStore",
    "EncryptedJsonFileAuthStore",
    "MirrorAuthStore",
    "InMemoryAuthStore",
    "RedisAuthStore",
    "GitAuthStore",
    "ObjectAuthStore",
    "TokenStorage",
    "JsonTokenStorage",
    "PlaintextJsonTokenStorage",
    "EncryptedJsonTokenStorage",
    "InMemoryTokenStorage",
    "TokenRecord",
    "ModelAliasMap",
    "ModelState",
    "QuotaState",
    "RequestContext",
    "AuthMaintainer",
    "list_auths",
    "get_auth",
    "save_auth",
    "delete_auth",
    "export_auths",
    "import_auths",
    "AuthMetrics",
    "AuthHooks",
    "NoopAuthHooks",
    "AuthManager",
    "WeightedSelector",
    "LatencySelector",
    "register_selector",
    "get_selector",
    "start_device_flow",
    "poll_device_flow",
    "mark_device_flow_authorized",
    "export_tokens",
    "import_tokens",
    "default_auth_store_dir",
    "default_token_store_dir",
]
