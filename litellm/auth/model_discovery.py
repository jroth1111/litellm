"""
Best-effort model discovery for OpenAI-compatible upstreams.

This is intentionally scoped: not every provider exposes a stable models list
API, and some require product-specific endpoints. We implement discovery where
the upstream speaks the OpenAI `/models` schema.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import asyncio
import httpx

from .core import AuthRecord, RequestContext


def _models_endpoint(base_url: str) -> str:
    base = (base_url or "").rstrip("/")
    if not base:
        raise ValueError("missing base_url for model discovery")
    parsed = urlparse(base)
    path = (parsed.path or "").rstrip("/")
    last = path.split("/")[-1] if path else ""
    # If caller already provided a versioned base (`.../v1` or `.../hf/v1`),
    # append `/models`. Otherwise default to OpenAI-style `/v1/models`.
    if last == "v1":
        return f"{base}/models"
    return f"{base}/v1/models"


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
    timeout_seconds: float = 10.0,
) -> Tuple[int, List[str], str]:
    """
    Returns (status_code, model_ids, raw_text) for diagnostics.
    """
    url = _models_endpoint(base_url)
    req_headers = {"Accept": "application/json"}
    req_headers.update(headers or {})

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        resp = await client.get(url, headers=req_headers)
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
    models: Tuple[str, ...]
    fetched_at: float
    source_auth_id: Optional[str] = None


class ModelDiscoveryCache:
    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._lock = threading.Lock()
        self._cache: Dict[Tuple[str, str], CachedModels] = {}

    def get(self, provider: str, base_url: str) -> Optional[CachedModels]:
        key = (provider.strip().lower(), (base_url or "").rstrip("/"))
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
        key = (entry.provider.strip().lower(), (entry.base_url or "").rstrip("/"))
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
) -> CachedModels:
    """
    Try each credential until one returns a valid models list.

    - 401/403: refresh once (if `refresh_record` provided), then retry.
    - 429: move to next credential.
    """
    cached = cache.get(provider, base_url)
    if cached is not None:
        return cached

    provider_key = provider.strip().lower()
    last_error: Optional[str] = None

    for rec in auth_records:
        if rec.provider.strip().lower() != provider_key:
            continue
        token = rec.metadata.get("access_token")
        if not token:
            continue

        ctx = RequestContext(model="")
        try:
            headers = prepare_headers({}, ctx, rec)
        except Exception as e:
            last_error = str(e)
            continue

        status, models, raw = await fetch_openai_compatible_models(
            base_url=base_url, headers=headers, timeout_seconds=timeout_seconds
        )
        if status == 200 and models:
            entry = CachedModels(
                provider=provider_key,
                base_url=(base_url or "").rstrip("/"),
                models=tuple(sorted(set(models))),
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
                status2, models2, raw2 = await fetch_openai_compatible_models(
                    base_url=base_url, headers=headers2, timeout_seconds=timeout_seconds
                )
                if status2 == 200 and models2:
                    entry = CachedModels(
                        provider=provider_key,
                        base_url=(base_url or "").rstrip("/"),
                        models=tuple(sorted(set(models2))),
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
