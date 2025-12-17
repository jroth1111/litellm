import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest
litellm = pytest.importorskip("litellm", reason="requires LiteLLM dependencies")
proxy_server = pytest.importorskip(
    "litellm.proxy.proxy_server",
    reason="requires LiteLLM proxy extras (pip install 'litellm[proxy]')",
)
auth_core = pytest.importorskip("litellm.auth.core")
auth_file_store = pytest.importorskip("litellm.auth.file_store")
auth_metrics = pytest.importorskip("litellm.auth.metrics")

from fastapi import FastAPI
from fastapi.testclient import TestClient

AuthRecord = auth_core.AuthRecord
JsonFileAuthStore = auth_file_store.JsonFileAuthStore
AuthMetrics = auth_metrics.AuthMetrics


class RecordingAuthHooks(auth_metrics.AuthHooks):
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def on_register(self, auth: AuthRecord) -> None:
        self.events.append(("register", auth.id))

    def on_update(self, auth: AuthRecord, reason: str = "") -> None:
        self.events.append(("update", auth.id, reason))

    def on_success(
        self,
        auth: AuthRecord,
        provider_model: str | None = None,
        retry_after: float | None = None,
        is_quota: bool = False,
    ) -> None:
        self.events.append(("success", auth.id, provider_model))

    def on_failure(
        self,
        auth: AuthRecord,
        provider_model: str | None = None,
        error: BaseException | None = None,
        retry_after: float | None = None,
        is_quota: bool = False,
        status_code: int | None = None,
    ) -> None:
        status_code = getattr(error, "status_code", None) if error is not None else None
        self.events.append(("failure", auth.id, provider_model, status_code))


@pytest.fixture(scope="function")
def client(tmp_path, monkeypatch, setup_and_teardown):
    monkeypatch.setenv("LITELLM_DONT_SHOW_FEEDBACK_BOX", "true")
    proxy_server.cleanup_router_config_variables()

    # Build a local auth store with two OpenAI subscription credentials.
    store_dir = tmp_path / "auth_store"
    store = JsonFileAuthStore(str(store_dir))
    now = datetime.now(timezone.utc)
    store.save(
        "default",
        AuthRecord(
            id="a1",
            provider="openai",
            metadata={"access_token": "tok1", "refresh_token": "rt1"},
            last_refreshed_at=now,
        ),
    )
    store.save(
        "default",
        AuthRecord(
            id="a2",
            provider="openai",
            metadata={"access_token": "tok2"},
            last_refreshed_at=now - timedelta(seconds=5),
        ),
    )

    # Minimal proxy config: a subscription-mode deployment with no static api_key.
    config_fp = tmp_path / "config.yaml"
    config_fp.write_text(
        "\n".join(
            [
                "model_list:",
                "  - model_name: openai-model",
                "    auth_mode: subscription",
                "    litellm_params:",
                "      model: openai/gpt-4o",
                "auth_settings:",
                "  enabled: true",
                "  store_backend: json",
                f'  store_dir: "{store_dir}"',
                "  namespace: default",
                "  mount_api: false",
                "  maintainer: false",
                "",
            ]
        ),
        encoding="utf-8",
    )

    asyncio.run(proxy_server.initialize(config=str(config_fp)))
    app = FastAPI()
    app.include_router(proxy_server.router)
    return TestClient(app), store


def test_same_request_rotation_on_429(client, monkeypatch):
    calls = []

    client, store = client
    router = proxy_server.llm_router
    hooks = RecordingAuthHooks()
    metrics = AuthMetrics()
    router.auth_hooks = hooks
    router.auth_metrics = metrics

    async def fake_acompletion(*_args, **kwargs):
        token = kwargs.get("api_key")
        calls.append(token)
        if token == "tok1":
            raise litellm.RateLimitError(
                message="rate limit",
                llm_provider="openai",
                model=str(kwargs.get("model", "")),
            )
        return litellm.ModelResponse(
            model=str(kwargs.get("model", "")),
            choices=[{"message": {"role": "assistant", "content": "ok"}}],
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["choices"][0]["message"]["content"] == "ok"
    assert calls == ["tok1", "tok2"]
    assert hooks.events == [
        ("register", "a1"),
        ("failure", "a1", "openai/gpt-4o", 429),
        ("register", "a2"),
        ("success", "a2", "openai/gpt-4o"),
    ]
    assert metrics.calls_by_provider["openai"] == 1
    assert metrics.quota_hits_by_provider["openai"] == 1
    # ensure failure state is persisted
    updated_a1 = store.get("default", "a1")
    assert updated_a1 is not None
    assert updated_a1.unavailable is True
    assert updated_a1.next_retry_after is not None
    assert "openai/gpt-4o" in updated_a1.model_states


def test_unhealthy_auth_is_skipped_on_next_request(client, monkeypatch):
    client, store = client

    calls = []

    async def fake_acompletion(*_args, **kwargs):
        token = kwargs.get("api_key")
        calls.append(token)
        if token == "tok1":
            raise litellm.RateLimitError(
                message="rate limit",
                llm_provider="openai",
                model=str(kwargs.get("model", "")),
            )
        return litellm.ModelResponse(
            model=str(kwargs.get("model", "")),
            choices=[{"message": {"role": "assistant", "content": "ok"}}],
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    resp1 = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )
    assert resp1.status_code == 200, resp1.text
    assert calls == ["tok1", "tok2"]

    updated_a1 = store.get("default", "a1")
    assert updated_a1 is not None
    assert updated_a1.unavailable is True
    assert updated_a1.next_retry_after is not None

    calls.clear()
    resp2 = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )
    assert resp2.status_code == 200, resp2.text
    assert resp2.json()["choices"][0]["message"]["content"] == "ok"
    assert calls == ["tok2"]


def test_same_request_refresh_on_401_then_retry(client, monkeypatch):
    """
    If the active subscription token returns 401/403, Router should attempt a refresh once
    and retry the same request with the refreshed token before rotating to a different account.
    """
    calls = []

    # Patch the active adapter to refresh access_token.
    client, store = client
    router = proxy_server.llm_router
    hooks = RecordingAuthHooks()
    router.auth_hooks = hooks
    strategy = router.auth_strategies["openai"]

    def fake_refresh(auth, _ctx):
        updated = auth.clone()
        updated.metadata["access_token"] = "tok1-refreshed"
        return updated

    monkeypatch.setattr(strategy, "refresh", fake_refresh)

    async def fake_acompletion(*_args, **kwargs):
        token = kwargs.get("api_key")
        calls.append(token)
        if token == "tok1":
            raise litellm.AuthenticationError(
                message="expired",
                llm_provider="openai",
                model=str(kwargs.get("model", "")),
            )
        return litellm.ModelResponse(
            model=str(kwargs.get("model", "")),
            choices=[{"message": {"role": "assistant", "content": "ok"}}],
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == "ok"
    assert calls == ["tok1", "tok1-refreshed"]
    assert hooks.events == [
        ("register", "a1"),
        ("update", "a1", "refresh"),
        ("success", "a1", "openai/gpt-4o"),
    ]
    refreshed = store.get("default", "a1")
    assert refreshed is not None
    assert refreshed.metadata.get("access_token") == "tok1-refreshed"


def test_proactive_refresh_before_expiry(client, monkeypatch):
    """
    If token expiry is near (within refresh lead), Router should proactively refresh
    before making the upstream call.
    """
    calls = []

    client, store = client
    router = proxy_server.llm_router
    hooks = RecordingAuthHooks()
    router.auth_hooks = hooks
    strategy = router.auth_strategies["openai"]

    def fake_expiration(_auth):
        return datetime.now(timezone.utc) + timedelta(seconds=1)

    def fake_refresh_lead(_auth):
        return timedelta(minutes=10)

    def fake_refresh(auth, _ctx):
        updated = auth.clone()
        updated.metadata["access_token"] = "tok1-proactive"
        updated.last_refreshed_at = datetime.now(timezone.utc)
        return updated

    monkeypatch.setattr(strategy, "expiration", fake_expiration)
    monkeypatch.setattr(strategy, "refresh_lead", fake_refresh_lead)
    monkeypatch.setattr(strategy, "refresh", fake_refresh)

    async def fake_acompletion(*_args, **kwargs):
        token = kwargs.get("api_key")
        calls.append(token)
        return litellm.ModelResponse(
            model=str(kwargs.get("model", "")),
            choices=[{"message": {"role": "assistant", "content": "ok"}}],
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == "ok"
    assert calls == ["tok1-proactive"]
    assert hooks.events == [
        ("register", "a1"),
        ("update", "a1", "refresh"),
        ("success", "a1", "openai/gpt-4o"),
    ]
    refreshed = store.get("default", "a1")
    assert refreshed is not None
    assert refreshed.metadata.get("access_token") == "tok1-proactive"


def test_rotate_when_refresh_fails(client, monkeypatch):
    """
    If refresh fails, Router should rotate to the next auth record within the same request.
    """
    calls = []

    client, _store = client
    router = proxy_server.llm_router
    strategy = router.auth_strategies["openai"]

    def fake_refresh(_auth, _ctx):
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(strategy, "refresh", fake_refresh)

    async def fake_acompletion(*_args, **kwargs):
        token = kwargs.get("api_key")
        calls.append(token)
        if token == "tok1":
            raise litellm.AuthenticationError(
                message="expired",
                llm_provider="openai",
                model=str(kwargs.get("model", "")),
            )
        return litellm.ModelResponse(
            model=str(kwargs.get("model", "")),
            choices=[{"message": {"role": "assistant", "content": "ok"}}],
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == "ok"
    assert calls == ["tok1", "tok2"]
