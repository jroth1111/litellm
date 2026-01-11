from __future__ import annotations

from litellm.canonical import (
    anthropic_request_to_canonical,
    anthropic_response_to_canonical,
    canonical_to_anthropic_request,
    canonical_to_anthropic_response,
    canonical_to_openai_request,
    canonical_to_openai_response,
    openai_request_to_canonical,
    openai_response_to_canonical,
)


def test_openai_request_roundtrip():
    request = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Hi",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"},
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
        "temperature": 0.2,
        "stream": False,
    }
    canonical = openai_request_to_canonical(request)
    roundtrip = canonical_to_openai_request(canonical)
    assert roundtrip["model"] == "gpt-4o"
    assert roundtrip["messages"][0]["role"] == "user"
    assert roundtrip["messages"][0]["content"] == "Hello"
    assert roundtrip["tools"][0]["function"]["name"] == "get_weather"
    assert roundtrip["temperature"] == 0.2


def test_anthropic_request_roundtrip():
    request = {
        "model": "claude-3-5-sonnet-20241022",
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {"type": "tool_use", "id": "tool_1", "name": "search", "input": {"q": "SF"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "Sunny",
                    }
                ],
            },
        ],
        "tools": [{"name": "search", "description": "Search", "input_schema": {"type": "object"}}],
        "max_tokens": 128,
    }
    canonical = anthropic_request_to_canonical(request)
    roundtrip = canonical_to_anthropic_request(canonical)
    assert roundtrip["model"] == "claude-3-5-sonnet-20241022"
    assert "system" in roundtrip and "helpful" in roundtrip["system"]
    assert roundtrip["messages"][0]["role"] == "user"
    assert roundtrip["messages"][0]["content"][0]["text"] == "Hello"
    assert roundtrip["tools"][0]["name"] == "search"
    assert roundtrip["max_tokens"] == 128


def test_openai_response_to_canonical():
    response = {
        "id": "chatcmpl_123",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Hello",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{\"q\":\"SF\"}"},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    canonical = openai_response_to_canonical(response)
    types = [block.get("type") for block in canonical["content"]]
    assert "text" in types
    assert "tool_use" in types


def test_anthropic_response_to_canonical():
    """Test converting Anthropic response to canonical format."""
    response = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [
            {"type": "text", "text": "Hello, I can help you with that."},
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {"city": "San Francisco"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 25},
    }
    canonical = anthropic_response_to_canonical(response)

    # Check content blocks
    assert len(canonical["content"]) == 2
    assert canonical["content"][0]["type"] == "text"
    assert canonical["content"][0]["text"] == "Hello, I can help you with that."
    assert canonical["content"][1]["type"] == "tool_use"
    assert canonical["content"][1]["name"] == "get_weather"
    assert canonical["content"][1]["input"] == {"city": "San Francisco"}

    # Check usage mapping
    assert canonical["usage"]["prompt_tokens"] == 10
    assert canonical["usage"]["completion_tokens"] == 25
    assert canonical["usage"]["total_tokens"] == 35

    # Check stop reason and model
    assert canonical["stop_reason"] == "tool_use"
    assert canonical["model"] == "claude-3-5-sonnet-20241022"


def test_canonical_to_openai_response():
    """Test converting canonical response to OpenAI format."""
    canonical_response = {
        "content": [
            {"type": "text", "text": "Here is the weather:"},
            {
                "type": "tool_use",
                "id": "call_abc123",
                "name": "get_weather",
                "input": {"city": "NYC"},
            },
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 30, "total_tokens": 45},
        "stop_reason": "tool_use",
        "model": "gpt-4o",
    }
    openai_response = canonical_to_openai_response(canonical_response)

    # Check structure
    assert openai_response["object"] == "chat.completion"
    assert openai_response["model"] == "gpt-4o"
    assert len(openai_response["choices"]) == 1

    # Check message
    message = openai_response["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert message["content"] == "Here is the weather:"

    # Check tool calls
    assert "tool_calls" in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["type"] == "function"
    assert message["tool_calls"][0]["function"]["name"] == "get_weather"

    # Check finish reason mapping
    assert openai_response["choices"][0]["finish_reason"] == "tool_calls"

    # Check usage
    assert openai_response["usage"]["prompt_tokens"] == 15
    assert openai_response["usage"]["completion_tokens"] == 30
    assert openai_response["usage"]["total_tokens"] == 45


def test_canonical_to_anthropic_response():
    """Test converting canonical response to Anthropic format."""
    canonical_response = {
        "content": [
            {"type": "text", "text": "Hello!"},
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        "stop_reason": "end_turn",
        "model": "claude-3-5-sonnet-20241022",
    }
    anthropic_response = canonical_to_anthropic_response(canonical_response)

    # Check structure
    assert anthropic_response["type"] == "message"
    assert anthropic_response["role"] == "assistant"
    assert anthropic_response["model"] == "claude-3-5-sonnet-20241022"

    # Check content
    assert len(anthropic_response["content"]) == 1
    assert anthropic_response["content"][0]["type"] == "text"
    assert anthropic_response["content"][0]["text"] == "Hello!"

    # Check usage mapping
    assert anthropic_response["usage"]["input_tokens"] == 5
    assert anthropic_response["usage"]["output_tokens"] == 10

    # Check stop reason
    assert anthropic_response["stop_reason"] == "end_turn"


def test_openai_to_anthropic_cross_format():
    """Test full cross-format conversion: OpenAI -> Canonical -> Anthropic."""
    openai_response = {
        "id": "chatcmpl_test",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "The weather is sunny.",
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    # Convert OpenAI -> Canonical -> Anthropic
    canonical = openai_response_to_canonical(openai_response)
    anthropic = canonical_to_anthropic_response(canonical)

    # Verify Anthropic format
    assert anthropic["type"] == "message"
    assert anthropic["role"] == "assistant"
    assert anthropic["content"][0]["text"] == "The weather is sunny."
    assert anthropic["usage"]["input_tokens"] == 10
    assert anthropic["usage"]["output_tokens"] == 5


def test_anthropic_to_openai_cross_format():
    """Test full cross-format conversion: Anthropic -> Canonical -> OpenAI."""
    anthropic_response = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [
            {"type": "text", "text": "I found the information."},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 20, "output_tokens": 8},
    }

    # Convert Anthropic -> Canonical -> OpenAI
    canonical = anthropic_response_to_canonical(anthropic_response)
    openai = canonical_to_openai_response(canonical)

    # Verify OpenAI format
    assert openai["object"] == "chat.completion"
    assert openai["choices"][0]["message"]["content"] == "I found the information."
    assert openai["choices"][0]["finish_reason"] == "stop"
    assert openai["usage"]["prompt_tokens"] == 20
    assert openai["usage"]["completion_tokens"] == 8


def test_tool_use_cross_format():
    """Test tool use conversion across formats."""
    anthropic_response = {
        "id": "msg_tool",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [
            {"type": "text", "text": "Let me check that."},
            {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "search",
                "input": {"query": "weather"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 15, "output_tokens": 20},
    }

    # Anthropic -> Canonical
    canonical = anthropic_response_to_canonical(anthropic_response)
    assert len(canonical["content"]) == 2
    assert canonical["content"][1]["type"] == "tool_use"
    assert canonical["content"][1]["name"] == "search"

    # Canonical -> OpenAI
    openai = canonical_to_openai_response(canonical)
    assert "tool_calls" in openai["choices"][0]["message"]
    assert openai["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "search"
    assert openai["choices"][0]["finish_reason"] == "tool_calls"


def test_empty_response_handling():
    """Test handling of empty/minimal responses."""
    # Empty canonical response
    canonical = {"content": []}
    openai = canonical_to_openai_response(canonical)
    anthropic = canonical_to_anthropic_response(canonical)

    assert openai["choices"][0]["message"]["content"] is None
    assert anthropic["content"] == []


def test_stop_reason_mapping():
    """Test stop reason mapping between formats."""
    test_cases = [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_calls"),
    ]

    for anthropic_reason, expected_openai_reason in test_cases:
        canonical = {
            "content": [{"type": "text", "text": "test"}],
            "stop_reason": anthropic_reason,
            "usage": {},
        }
        openai = canonical_to_openai_response(canonical)
        assert openai["choices"][0]["finish_reason"] == expected_openai_reason, \
            f"Expected {expected_openai_reason} for {anthropic_reason}"
