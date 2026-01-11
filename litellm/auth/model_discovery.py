"""
Best-effort model discovery for upstreams that expose a models list.

This is intentionally scoped: not every provider exposes a stable models list
API, and some require product-specific endpoints. We implement discovery where
the upstream speaks a JSON list schema compatible with OpenAI-style payloads.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import asyncio
import httpx

from .core import AuthRecord, RequestContext


@dataclass(frozen=True)
class ModelCapabilities:
    """Capabilities of a model."""

    vision: bool = False
    tool_use: bool = False
    streaming: bool = True
    json_mode: bool = False
    function_calling: bool = False


@dataclass(frozen=True)
class ModelInfo:
    """
    Rich model information including metadata.

    Matches the ModelInfo schema from the requirements:
    - id: Model identifier
    - provider: Provider name
    - display_name: Human-readable name
    - aliases: Alternative names
    - context_window: Max input tokens
    - max_output_tokens: Max output tokens
    - capabilities: Feature flags
    """

    id: str
    provider: str
    display_name: str = ""
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    created: Optional[int] = None
    owned_by: Optional[str] = None

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI /v1/models response format."""
        result: Dict[str, Any] = {
            "id": self.id,
            "object": "model",
            "created": self.created or int(time.time()),
            "owned_by": self.owned_by or self.provider,
        }
        # Add extended metadata if available
        if self.context_window:
            result["context_window"] = self.context_window
        if self.max_output_tokens:
            result["max_output_tokens"] = self.max_output_tokens
        return result

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic /v1/models response format."""
        result: Dict[str, Any] = {
            "id": self.id,
            "type": "model",
            "display_name": self.display_name or self.id,
            "created_at": (
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.created))
                if self.created
                else None
            ),
        }
        if self.context_window:
            result["context_window"] = self.context_window
        if self.max_output_tokens:
            result["max_output_tokens"] = self.max_output_tokens
        return result


# Known model metadata (static fallback for providers without discovery)
KNOWN_MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    # Anthropic models
    "claude-3-5-sonnet-20241022": {
        "context_window": 200000,
        "max_output_tokens": 8192,
        "capabilities": {"vision": True, "tool_use": True, "streaming": True},
    },
    "claude-3-5-haiku-20241022": {
        "context_window": 200000,
        "max_output_tokens": 8192,
        "capabilities": {"vision": True, "tool_use": True, "streaming": True},
    },
    "claude-3-opus-20240229": {
        "context_window": 200000,
        "max_output_tokens": 4096,
        "capabilities": {"vision": True, "tool_use": True, "streaming": True},
    },
    "claude-sonnet-4-20250514": {
        "context_window": 200000,
        "max_output_tokens": 16384,
        "capabilities": {"vision": True, "tool_use": True, "streaming": True},
    },
    "claude-opus-4-20250514": {
        "context_window": 200000,
        "max_output_tokens": 32768,
        "capabilities": {"vision": True, "tool_use": True, "streaming": True},
    },
    # OpenAI models
    "gpt-4o": {
        "context_window": 128000,
        "max_output_tokens": 16384,
        "capabilities": {"vision": True, "tool_use": True, "streaming": True, "json_mode": True},
    },
    "gpt-4o-mini": {
        "context_window": 128000,
        "max_output_tokens": 16384,
        "capabilities": {"vision": True, "tool_use": True, "streaming": True, "json_mode": True},
    },
    "gpt-4-turbo": {
        "context_window": 128000,
        "max_output_tokens": 4096,
        "capabilities": {"vision": True, "tool_use": True, "streaming": True, "json_mode": True},
    },
    "gpt-3.5-turbo": {
        "context_window": 16385,
        "max_output_tokens": 4096,
        "capabilities": {"vision": False, "tool_use": True, "streaming": True, "json_mode": True},
    },
    "o1": {
        "context_window": 200000,
        "max_output_tokens": 100000,
        "capabilities": {"vision": True, "tool_use": True, "streaming": True},
    },
    "o1-mini": {
        "context_window": 128000,
        "max_output_tokens": 65536,
        "capabilities": {"vision": False, "tool_use": True, "streaming": True},
    },
    # Gemini models
    "gemini-2.5-pro": {
        "context_window": 1000000,
        "max_output_tokens": 65536,
        "capabilities": {"vision": True, "tool_use": True, "streaming": True},
    },
    "gemini-2.0-flash": {
        "context_window": 1000000,
        "max_output_tokens": 8192,
        "capabilities": {"vision": True, "tool_use": True, "streaming": True},
    },
    "gemini-1.5-pro": {
        "context_window": 2000000,
        "max_output_tokens": 8192,
        "capabilities": {"vision": True, "tool_use": True, "streaming": True},
    },
    "gemini-1.5-flash": {
        "context_window": 1000000,
        "max_output_tokens": 8192,
        "capabilities": {"vision": True, "tool_use": True, "streaming": True},
    },
}


def get_model_metadata(model_id: str) -> Optional[Dict[str, Any]]:
    """Look up known metadata for a model ID."""
    # Direct lookup
    if model_id in KNOWN_MODEL_METADATA:
        return KNOWN_MODEL_METADATA[model_id]
    # Try without provider prefix
    if "/" in model_id:
        base_id = model_id.split("/", 1)[1]
        if base_id in KNOWN_MODEL_METADATA:
            return KNOWN_MODEL_METADATA[base_id]
    # Try partial match for versioned models
    for known_id, meta in KNOWN_MODEL_METADATA.items():
        if model_id.startswith(known_id.rsplit("-", 1)[0]):
            return meta
    return None


def infer_provider_from_model_id(model_id: str) -> Optional[str]:
    """
    Best-effort provider inference for known model IDs.
    """
    if not isinstance(model_id, str):
        return None
    lowered = model_id.lower()
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith(("gpt-", "chatgpt", "o1", "o3")):
        return "openai"
    if lowered.startswith("gemini"):
        return "gemini"
    if lowered.startswith("qwen"):
        return "qwen"
    if lowered.startswith("cursor"):
        return "cursor"
    if lowered.startswith("copilot"):
        return "github_copilot"
    return None


def static_models_for_provider(provider: str) -> List[str]:
    """
    Return a static fallback list of models for providers without discovery APIs.
    """
    provider_key = (provider or "").strip().lower()
    if not provider_key:
        return []
    models: List[str] = []
    for model_id in KNOWN_MODEL_METADATA.keys():
        inferred = infer_provider_from_model_id(model_id)
        if inferred == provider_key:
            models.append(model_id)
    return models


def _parse_model_info(item: Dict[str, Any], provider: str) -> ModelInfo:
    """Parse a model item from API response into ModelInfo."""
    model_id = item.get("id") or item.get("name") or ""

    # Get known metadata or use defaults
    known = get_model_metadata(model_id) or {}

    # Parse capabilities
    caps_data = known.get("capabilities", {})
    if isinstance(caps_data, dict):
        capabilities = ModelCapabilities(
            vision=caps_data.get("vision", False),
            tool_use=caps_data.get("tool_use", False),
            streaming=caps_data.get("streaming", True),
            json_mode=caps_data.get("json_mode", False),
            function_calling=caps_data.get("function_calling", caps_data.get("tool_use", False)),
        )
    else:
        capabilities = ModelCapabilities()

    return ModelInfo(
        id=model_id,
        provider=provider,
        display_name=item.get("display_name") or item.get("name") or model_id,
        context_window=item.get("context_window") or known.get("context_window"),
        max_output_tokens=item.get("max_output_tokens") or known.get("max_output_tokens"),
        capabilities=capabilities,
        created=item.get("created"),
        owned_by=item.get("owned_by"),
    )


def _models_endpoint(base_url: str, api_version: str = "v1") -> str:
    base = (base_url or "").rstrip("/")
    if not base:
        raise ValueError("missing base_url for model discovery")
    parsed = urlparse(base)
    path = (parsed.path or "").rstrip("/")
    last = path.split("/")[-1] if path else ""
    version = (api_version or "v1").strip().lower()
    # If caller already provided a versioned base (`.../v1`, `.../v1beta`, etc),
    # append `/models`. Otherwise default to `/{version}/models`.
    if last == version:
        return f"{base}/models"
    return f"{base}/{version}/models"


def _parse_models_payload(payload: Any) -> List[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, list):
        out: List[str] = []
        for item in data:
            if isinstance(item, dict):
                mid = item.get("id") or item.get("name")
                if isinstance(mid, str) and mid:
                    out.append(mid)
        return out
    models = payload.get("models")
    if isinstance(models, list):
        out = []
        for item in models:
            if isinstance(item, str) and item:
                out.append(item)
            elif isinstance(item, dict):
                mid = item.get("id") or item.get("name")
                if isinstance(mid, str) and mid:
                    out.append(mid)
        return out
    return []


async def fetch_openai_compatible_models(
    *,
    base_url: str,
    headers: Dict[str, str],
    api_version: str = "v1",
    params: Optional[Dict[str, str]] = None,
    timeout_seconds: float = 10.0,
) -> Tuple[int, List[str], str]:
    """
    Returns (status_code, model_ids, raw_text) for diagnostics.
    """
    url = _models_endpoint(base_url, api_version=api_version)
    req_headers = {"Accept": "application/json"}
    req_headers.update(headers or {})

    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.types.llms.custom_http import httpxSpecialProvider

    async_client = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.Oauth2Check,
        params={"timeout": httpx.Timeout(timeout_seconds)},
    )
    resp = await async_client.get(url, headers=req_headers, params=params)
    raw = resp.text
    try:
        payload = resp.json()
    except Exception:
        payload = None
    return resp.status_code, _parse_models_payload(payload), raw


@dataclass(frozen=True)
class CachedModels:
    provider: str
    base_url: str
    api_version: str
    models: Tuple[str, ...]
    fetched_at: float
    source_auth_id: Optional[str] = None


class ModelDiscoveryCache:
    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._lock = threading.Lock()
        self._cache: Dict[Tuple[str, str], CachedModels] = {}

    def get(
        self, provider: str, base_url: str, api_version: str
    ) -> Optional[CachedModels]:
        key = (
            provider.strip().lower(),
            (base_url or "").rstrip("/"),
            (api_version or "v1").strip().lower(),
        )
        now = time.time()
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if now - entry.fetched_at > self.ttl_seconds:
                self._cache.pop(key, None)
                return None
            return entry

    def set(self, entry: CachedModels) -> None:
        key = (
            entry.provider.strip().lower(),
            (entry.base_url or "").rstrip("/"),
            (entry.api_version or "v1").strip().lower(),
        )
        with self._lock:
            self._cache[key] = entry


DEFAULT_MODEL_DISCOVERY_CACHE = ModelDiscoveryCache(ttl_seconds=300)


async def discover_models_with_rotation(
    *,
    provider: str,
    base_url: str,
    auth_records: Iterable[AuthRecord],
    prepare_headers,
    refresh_record=None,
    timeout_seconds: float = 10.0,
    cache: ModelDiscoveryCache = DEFAULT_MODEL_DISCOVERY_CACHE,
    force_refresh: bool = False,
    api_version: str = "v1",
    extra_headers: Optional[Dict[str, str]] = None,
    normalize_model_id: Optional[Callable[[str], Optional[str]]] = None,
    params: Optional[Dict[str, str]] = None,
) -> CachedModels:
    """
    Try each credential until one returns a valid models list.

    - 401/403: refresh once (if `refresh_record` provided), then retry.
    - 429: move to next credential.
    """
    if not force_refresh:
        cached = cache.get(provider, base_url, api_version)
        if cached is not None:
            return cached

    provider_key = provider.strip().lower()
    last_error: Optional[str] = None

    def _normalize_models(models: List[str]) -> List[str]:
        if not normalize_model_id:
            return [m for m in models if isinstance(m, str) and m]
        normalized: List[str] = []
        for mid in models:
            if not isinstance(mid, str) or not mid:
                continue
            val = normalize_model_id(mid)
            if isinstance(val, str) and val:
                normalized.append(val)
        return normalized

    for rec in auth_records:
        if rec.provider.strip().lower() != provider_key:
            continue
        token = rec.resolve_secret()
        if not token:
            continue

        ctx = RequestContext(model="")
        try:
            headers = prepare_headers({}, ctx, rec)
        except Exception as e:
            last_error = str(e)
            continue
        if extra_headers:
            headers.update(extra_headers)

        status, models, raw = await fetch_openai_compatible_models(
            base_url=base_url,
            headers=headers,
            api_version=api_version,
            params=params,
            timeout_seconds=timeout_seconds,
        )
        normalized_models = _normalize_models(models)
        if status == 200 and normalized_models:
            entry = CachedModels(
                provider=provider_key,
                base_url=(base_url or "").rstrip("/"),
                api_version=(api_version or "v1").strip().lower(),
                models=tuple(sorted(set(normalized_models))),
                fetched_at=time.time(),
                source_auth_id=rec.id,
            )
            cache.set(entry)
            return entry

        if status in (401, 403) and refresh_record is not None:
            try:
                refreshed = await asyncio.to_thread(refresh_record, rec)
            except Exception as e:
                last_error = str(e)
                continue
            token2 = refreshed.metadata.get("access_token")
            if token2:
                try:
                    headers2 = prepare_headers({}, ctx, refreshed)
                except Exception as e:
                    last_error = str(e)
                    continue
                if extra_headers:
                    headers2.update(extra_headers)
                status2, models2, raw2 = await fetch_openai_compatible_models(
                    base_url=base_url,
                    headers=headers2,
                    api_version=api_version,
                    params=params,
                    timeout_seconds=timeout_seconds,
                )
                normalized_models2 = _normalize_models(models2)
                if status2 == 200 and normalized_models2:
                    entry = CachedModels(
                        provider=provider_key,
                        base_url=(base_url or "").rstrip("/"),
                        api_version=(api_version or "v1").strip().lower(),
                        models=tuple(sorted(set(normalized_models2))),
                        fetched_at=time.time(),
                        source_auth_id=refreshed.id,
                    )
                    cache.set(entry)
                    return entry
                last_error = f"{status2}: {raw2[:200]}"
                continue

        if status == 429:
            last_error = f"429: {raw[:200]}"
            continue

        last_error = f"{status}: {raw[:200]}"

    raise ValueError(
        f"model discovery failed for provider={provider_key} base_url={base_url}: {last_error or 'no usable credentials'}"
    )


def models_to_openai_response(
    models: Iterable[str],
    provider: str = "",
) -> Dict[str, Any]:
    """
    Convert a list of model IDs to OpenAI /v1/models response format.

    Args:
        models: Iterable of model ID strings
        provider: Provider name for metadata lookup

    Returns:
        OpenAI-compatible models list response
    """
    data: List[Dict[str, Any]] = []
    for model_id in models:
        info = _parse_model_info({"id": model_id}, provider)
        data.append(info.to_openai_format())

    return {
        "object": "list",
        "data": data,
    }


def models_to_anthropic_response(
    models: Iterable[str],
    provider: str = "",
) -> Dict[str, Any]:
    """
    Convert a list of model IDs to Anthropic /v1/models response format.

    Anthropic's models endpoint returns a different structure than OpenAI's.

    Args:
        models: Iterable of model ID strings
        provider: Provider name for metadata lookup

    Returns:
        Anthropic-compatible models list response
    """
    data: List[Dict[str, Any]] = []
    for model_id in models:
        info = _parse_model_info({"id": model_id}, provider)
        data.append(info.to_anthropic_format())

    return {
        "data": data,
        "has_more": False,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    }


def get_model_info_list(
    models: Iterable[str],
    provider: str = "",
) -> List[ModelInfo]:
    """
    Convert model IDs to a list of ModelInfo objects with metadata.

    Args:
        models: Iterable of model ID strings
        provider: Provider name

    Returns:
        List of ModelInfo objects
    """
    return [_parse_model_info({"id": mid}, provider) for mid in models]


class BackgroundModelRefresher:
    """
    Background thread that periodically refreshes the model discovery cache.

    This ensures model lists stay current even when no requests are made.
    The refresh interval should be shorter than the cache TTL to prevent
    stale cache entries.

    Usage:
        refresher = BackgroundModelRefresher(
            cache=DEFAULT_MODEL_DISCOVERY_CACHE,
            interval_seconds=240,  # Refresh every 4 minutes (TTL is 5 minutes)
        )
        # Register providers to refresh
        refresher.register_provider(
            provider="openai",
            base_url="https://api.openai.com",
            auth_records=lambda: auth_store.list("default"),
            prepare_headers=openai_adapter.prepare,
        )
        refresher.start()
        ...
        refresher.stop()
    """

    def __init__(
        self,
        cache: ModelDiscoveryCache = DEFAULT_MODEL_DISCOVERY_CACHE,
        interval_seconds: int = 240,
    ) -> None:
        self.cache = cache
        self.interval_seconds = max(30, interval_seconds)
        self._providers: Dict[str, Dict[str, Any]] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def register_provider(
        self,
        *,
        provider: str,
        base_url: str,
        auth_records: Callable[[], Iterable[AuthRecord]],
        prepare_headers: Callable,
        refresh_record: Optional[Callable] = None,
        timeout_seconds: float = 10.0,
        api_version: str = "v1",
        extra_headers: Optional[Dict[str, str]] = None,
        normalize_model_id: Optional[Callable[[str], Optional[str]]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Register a provider for background model refresh.

        Args:
            provider: Provider identifier (e.g., "openai", "anthropic")
            base_url: Base URL for the models endpoint
            auth_records: Callable that returns an iterable of AuthRecords
            prepare_headers: Function to prepare request headers
            refresh_record: Optional function to refresh expired credentials
            timeout_seconds: Request timeout
            api_version: API version string
            extra_headers: Additional headers to include
            normalize_model_id: Optional function to normalize model IDs
            params: Optional query parameters
        """
        key = f"{provider.lower()}:{base_url}"
        with self._lock:
            self._providers[key] = {
                "provider": provider,
                "base_url": base_url,
                "auth_records": auth_records,
                "prepare_headers": prepare_headers,
                "refresh_record": refresh_record,
                "timeout_seconds": timeout_seconds,
                "api_version": api_version,
                "extra_headers": extra_headers,
                "normalize_model_id": normalize_model_id,
                "params": params,
            }

    def unregister_provider(self, provider: str, base_url: str) -> None:
        """Remove a provider from background refresh."""
        key = f"{provider.lower()}:{base_url}"
        with self._lock:
            self._providers.pop(key, None)

    def start(self) -> None:
        """Start the background refresh thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background refresh thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        """Background thread main loop."""
        while not self._stop_event.is_set():
            try:
                self._refresh_all()
            except Exception:
                # Best-effort; swallow to avoid killing the thread
                pass
            self._stop_event.wait(self.interval_seconds)

    def _refresh_all(self) -> None:
        """Refresh all registered providers."""
        with self._lock:
            providers = list(self._providers.items())

        for key, config in providers:
            try:
                self._refresh_provider(config)
            except Exception:
                # Continue with other providers even if one fails
                pass

    def _refresh_provider(self, config: Dict[str, Any]) -> None:
        """Refresh a single provider's model cache."""
        # Get current auth records
        auth_records_fn = config["auth_records"]
        auth_records = list(auth_records_fn())
        if not auth_records:
            return

        # Run the async discovery in a new event loop
        async def _discover():
            return await discover_models_with_rotation(
                provider=config["provider"],
                base_url=config["base_url"],
                auth_records=auth_records,
                prepare_headers=config["prepare_headers"],
                refresh_record=config.get("refresh_record"),
                timeout_seconds=config.get("timeout_seconds", 10.0),
                cache=self.cache,
                force_refresh=True,  # Force refresh even if cached
                api_version=config.get("api_version", "v1"),
                extra_headers=config.get("extra_headers"),
                normalize_model_id=config.get("normalize_model_id"),
                params=config.get("params"),
            )

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_discover())
            finally:
                loop.close()
        except Exception:
            # Discovery may fail (no healthy credentials, network issues, etc.)
            # This is acceptable for background refresh - we'll try again next interval
            pass
