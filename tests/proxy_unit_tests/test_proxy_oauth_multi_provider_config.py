import asyncio
from datetime import datetime, timezone

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
            id="o1",
            provider="openai",
            metadata={"access_token": "openai-tok-1", "refresh_token": "openai-rt-1"},
            last_refreshed_at=now,
        ),
    )
    store.save(
        "default",
        AuthRecord(
            id="o2",
            provider="openai",
            metadata={"access_token": "openai-tok-2", "refresh_token": "openai-rt-2"},
            last_refreshed_at=now,
        ),
    )
    store.save(
        "default",
        AuthRecord(
            id="a1",
            provider="anthropic",
            metadata={"access_token": "anthropic-tok-1", "refresh_token": "anthropic-rt-1"},
            last_refreshed_at=now,
        ),
    )

    config_fp = tmp_path / "config.yaml"
    config_fp.write_text(
        "\n".join(
            [
                "model_list:",
                "  - model_name: openai-model",
                "    auth_mode: subscription",
                "    litellm_params:",
                "      model: openai/gpt-4o",
                "  - model_name: claude-model",
                "    auth_mode: subscription",
                "    litellm_params:",
                "      model: anthropic/claude-3-5-sonnet",
                "auth_settings:",
                "  enabled: true",
                "  store_backend: json",
                f'  store_dir: \"{store_dir}\"',
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


def test_multi_provider_config_uses_correct_auth_per_provider(client, monkeypatch):
    calls = []

    async def fake_acompletion(*_args, **kwargs):
        model = str(kwargs.get("model", ""))
        headers = kwargs.get("headers") or {}
        calls.append(
            (
                model,
                kwargs.get("api_key"),
                headers.get("Authorization"),
            )
        )
        return litellm.ModelResponse(
            model=model,
            choices=[{"message": {"role": "assistant", "content": "ok"}}],
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    resp_openai = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )
    assert resp_openai.status_code == 200, resp_openai.text

    resp_anthropic = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        },
    )
    assert resp_anthropic.status_code == 200, resp_anthropic.text

    assert calls[0][0] == "openai/gpt-4o"
    assert calls[0][1] in ("openai-tok-1", "openai-tok-2")
    assert calls[1][0] == "anthropic/claude-3-5-sonnet"
    assert calls[1][2] == "Bearer anthropic-tok-1"


def test_auth_rotation_is_scoped_to_provider(client, monkeypatch):
    calls = []

    async def fake_acompletion(*_args, **kwargs):
        model = str(kwargs.get("model", ""))
        token = kwargs.get("api_key")
        calls.append((model, token))
        if model == "openai/gpt-4o" and token == "openai-tok-1":
            raise litellm.RateLimitError(
                message="rate limit",
                llm_provider="openai",
                model=model,
            )
        return litellm.ModelResponse(
            model=model,
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
    # Rotation should stay within OpenAI accounts, not fall back to the Anthropic auth record.
    assert calls == [
        ("openai/gpt-4o", "openai-tok-1"),
        ("openai/gpt-4o", "openai-tok-2"),
    ]
