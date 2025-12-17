import asyncio
from datetime import datetime, timedelta, timezone

import pytest

litellm = pytest.importorskip("litellm", reason="requires LiteLLM dependencies")
proxy_server = pytest.importorskip(
    "litellm.proxy.proxy_server",
    reason="requires LiteLLM proxy extras (pip install 'litellm[proxy]')",
)
auth_core = pytest.importorskip("litellm.auth.core")
auth_file_store = pytest.importorskip("litellm.auth.file_store")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.types.utils import AnthropicMessagesResponse

AuthRecord = auth_core.AuthRecord
JsonFileAuthStore = auth_file_store.JsonFileAuthStore


def _make_client(*, tmp_path, monkeypatch, fake_handler):
    monkeypatch.setenv("LITELLM_DONT_SHOW_FEEDBACK_BOX", "true")
    proxy_server.cleanup_router_config_variables()

    monkeypatch.setattr(litellm, "anthropic_messages", fake_handler)

    store_dir = tmp_path / "auth_store"
    store = JsonFileAuthStore(str(store_dir))
    now = datetime.now(timezone.utc)
    store.save(
        "default",
        AuthRecord(
            id="a1",
            provider="anthropic",
            metadata={"access_token": "tok1", "refresh_token": "rt1"},
            last_refreshed_at=now,
        ),
    )
    store.save(
        "default",
        AuthRecord(
            id="a2",
            provider="anthropic",
            metadata={"access_token": "tok2", "refresh_token": "rt2"},
            last_refreshed_at=now - timedelta(seconds=5),
        ),
    )

    config_fp = tmp_path / "config.yaml"
    config_fp.write_text(
        "\n".join(
            [
                "model_list:",
                "  - model_name: claude-model",
                "    auth_mode: subscription",
                "    litellm_params:",
                "      model: anthropic/claude-3-5-sonnet",
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
    return TestClient(app)


def test_messages_same_request_rotation_on_429(tmp_path, monkeypatch, setup_and_teardown):
    calls = []

    async def fake_messages(*_args, **kwargs):
        headers = (kwargs.get("headers") or {}).copy()
        token = headers.get("Authorization") or ""
        calls.append(token)
        if token.endswith("tok1"):
            raise litellm.RateLimitError(
                message="rate limit",
                llm_provider="anthropic",
                model=str(kwargs.get("model", "")),
            )
        return AnthropicMessagesResponse(
            id="msg_1",
            type="message",
            role="assistant",
            content=[{"type": "text", "text": "ok"}],
            model=str(kwargs.get("model", "")),
            stop_reason="end_turn",
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    client = _make_client(tmp_path=tmp_path, monkeypatch=monkeypatch, fake_handler=fake_messages)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["content"][0]["text"] == "ok"
    assert calls == ["Bearer tok1", "Bearer tok2"]


def test_messages_refresh_on_401_then_retry(tmp_path, monkeypatch, setup_and_teardown):
    calls = []

    async def fake_messages(*_args, **kwargs):
        headers = (kwargs.get("headers") or {}).copy()
        token = headers.get("Authorization") or ""
        calls.append(token)
        if token.endswith("tok1"):
            raise litellm.AuthenticationError(
                message="expired",
                llm_provider="anthropic",
                model=str(kwargs.get("model", "")),
            )
        return AnthropicMessagesResponse(
            id="msg_2",
            type="message",
            role="assistant",
            content=[{"type": "text", "text": "ok"}],
            model=str(kwargs.get("model", "")),
            stop_reason="end_turn",
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    client = _make_client(tmp_path=tmp_path, monkeypatch=monkeypatch, fake_handler=fake_messages)

    router = proxy_server.llm_router
    strategy = router.auth_strategies["anthropic"]

    def fake_refresh(auth, _ctx):
        updated = auth.clone()
        updated.metadata["access_token"] = "tok1-refreshed"
        return updated

    monkeypatch.setattr(strategy, "refresh", fake_refresh)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["content"][0]["text"] == "ok"
    assert calls == ["Bearer tok1", "Bearer tok1-refreshed"]


def test_messages_rotate_when_refresh_fails(tmp_path, monkeypatch, setup_and_teardown):
    calls = []

    async def fake_messages(*_args, **kwargs):
        headers = (kwargs.get("headers") or {}).copy()
        token = headers.get("Authorization") or ""
        calls.append(token)
        if token.endswith("tok1"):
            raise litellm.AuthenticationError(
                message="expired",
                llm_provider="anthropic",
                model=str(kwargs.get("model", "")),
            )
        return AnthropicMessagesResponse(
            id="msg_3",
            type="message",
            role="assistant",
            content=[{"type": "text", "text": "ok"}],
            model=str(kwargs.get("model", "")),
            stop_reason="end_turn",
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    client = _make_client(tmp_path=tmp_path, monkeypatch=monkeypatch, fake_handler=fake_messages)

    router = proxy_server.llm_router
    strategy = router.auth_strategies["anthropic"]

    def fake_refresh(_auth, _ctx):
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(strategy, "refresh", fake_refresh)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["content"][0]["text"] == "ok"
    assert calls == ["Bearer tok1", "Bearer tok2"]

