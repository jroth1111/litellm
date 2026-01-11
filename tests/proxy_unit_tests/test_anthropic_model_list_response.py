import pytest

utils = pytest.importorskip(
    "litellm.proxy.utils", reason="requires LiteLLM proxy extras"
)


def test_anthropic_model_list_default_limit():
    model_ids = [f"model-{idx}" for idx in range(1, 26)]
    payload = utils.build_anthropic_model_list_response(model_ids)

    assert len(payload["data"]) == 20
    assert payload["first_id"] == "model-1"
    assert payload["last_id"] == "model-20"
    assert payload["has_more"] is True


def test_anthropic_model_list_after_before_limit():
    model_ids = ["m0", "m1", "m2", "m3", "m4"]
    payload = utils.build_anthropic_model_list_response(
        model_ids, after_id="m1", before_id="m4", limit=2
    )
    returned_ids = [item["id"] for item in payload["data"]]
    assert returned_ids == ["m2", "m3"]
    assert payload["has_more"] is False
