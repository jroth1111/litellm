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

from fastapi import FastAPI
from fastapi.testclient import TestClient

AuthRecord = auth_core.AuthRecord
JsonFileAuthStore = auth_file_store.JsonFileAuthStore


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
            metadata={"access_token": "tok1"},
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
    return TestClient(app)


def test_same_request_rotation_on_429(client, monkeypatch):
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
