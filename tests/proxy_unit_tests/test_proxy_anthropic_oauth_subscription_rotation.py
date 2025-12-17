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

AuthRecord = auth_core.AuthRecord
JsonFileAuthStore = auth_file_store.JsonFileAuthStore


@pytest.fixture(scope="function")
def client(tmp_path, monkeypatch, setup_and_teardown):
    monkeypatch.setenv("LITELLM_DONT_SHOW_FEEDBACK_BOX", "true")
    proxy_server.cleanup_router_config_variables()

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


def test_same_request_rotation_on_429(client, monkeypatch):
    calls = []

    async def fake_acompletion(*_args, **kwargs):
        headers = kwargs.get("headers") or {}
        token = headers.get("Authorization") or ""
        calls.append(token)
        if token.endswith("tok1"):
            raise litellm.RateLimitError(
                message="rate limit",
                llm_provider="anthropic",
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
            "model": "claude-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == "ok"
    assert calls == ["Bearer tok1", "Bearer tok2"]


def test_same_request_refresh_on_401_then_retry(client, monkeypatch):
    calls = []
    router = proxy_server.llm_router
    strategy = router.auth_strategies["anthropic"]

    def fake_refresh(auth, _ctx):
        updated = auth.clone()
        updated.metadata["access_token"] = "tok1-refreshed"
        return updated

    monkeypatch.setattr(strategy, "refresh", fake_refresh)

    async def fake_acompletion(*_args, **kwargs):
        headers = kwargs.get("headers") or {}
        token = headers.get("Authorization") or ""
        calls.append(token)
        if token.endswith("tok1"):
            raise litellm.AuthenticationError(
                message="expired",
                llm_provider="anthropic",
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
            "model": "claude-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == "ok"
    assert calls == ["Bearer tok1", "Bearer tok1-refreshed"]


def test_rotate_when_refresh_fails(client, monkeypatch):
    calls = []
    router = proxy_server.llm_router
    strategy = router.auth_strategies["anthropic"]

    def fake_refresh(_auth, _ctx):
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(strategy, "refresh", fake_refresh)

    async def fake_acompletion(*_args, **kwargs):
        headers = kwargs.get("headers") or {}
        token = headers.get("Authorization") or ""
        calls.append(token)
        if token.endswith("tok1"):
            raise litellm.AuthenticationError(
                message="expired",
                llm_provider="anthropic",
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
            "model": "claude-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == "ok"
    assert calls == ["Bearer tok1", "Bearer tok2"]
