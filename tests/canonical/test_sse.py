"""Tests for canonical streaming adapters."""
from __future__ import annotations

from litellm.canonical.sse import (
    AnthropicStreamToCanonicalAdapter,
    CanonicalStreamToAnthropicAdapter,
    CanonicalStreamToOpenAIAdapter,
    OpenAIStreamToCanonicalAdapter,
)


class TestOpenAIStreamToCanonicalAdapter:
    """Tests for OpenAI -> Canonical streaming conversion."""

    def test_basic_text_stream(self):
        """Test converting basic text streaming chunks."""
        adapter = OpenAIStreamToCanonicalAdapter()

        # First chunk with role (may also emit empty text delta)
        chunk1 = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}],
        }
        events = adapter.feed(chunk1)
        # Returns message_start + content_block_delta for empty content
        assert len(events) >= 1
        assert events[0]["type"] == "message_start"
        assert events[0]["data"]["role"] == "assistant"

        # Content chunk
        chunk2 = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": "Hello"}}],
        }
        events = adapter.feed(chunk2)
        assert len(events) == 1
        assert events[0]["type"] == "content_block_delta"
        assert events[0]["data"]["delta"]["text"] == "Hello"

        # Final chunk
        chunk3 = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        events = adapter.feed(chunk3)
        assert len(events) == 1
        assert events[0]["type"] == "message_stop"
        assert events[0]["data"]["stop_reason"] == "stop"

    def test_tool_calls_stream(self):
        """Test converting tool call streaming chunks."""
        adapter = OpenAIStreamToCanonicalAdapter()

        # Tool call chunk
        chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_123",
                                "function": {"name": "get_weather", "arguments": '{"city"'},
                            }
                        ]
                    },
                }
            ],
        }
        events = adapter.feed(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "content_block_delta"
        assert events[0]["data"]["delta"]["type"] == "tool_use_delta"


class TestAnthropicStreamToCanonicalAdapter:
    """Tests for Anthropic -> Canonical streaming conversion."""

    def test_message_start(self):
        """Test message_start event conversion."""
        adapter = AnthropicStreamToCanonicalAdapter()

        events = adapter.feed(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-5-sonnet-20241022",
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
        )

        assert len(events) == 1
        assert events[0]["type"] == "message_start"
        assert events[0]["data"]["role"] == "assistant"
        assert events[0]["data"]["model"] == "claude-3-5-sonnet-20241022"

    def test_content_block_text(self):
        """Test text content block events."""
        adapter = AnthropicStreamToCanonicalAdapter()

        # Content block start
        events = adapter.feed(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        )
        assert len(events) == 1
        assert events[0]["type"] == "content_block_start"
        assert events[0]["data"]["type"] == "text"

        # Text delta
        events = adapter.feed(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
        )
        assert len(events) == 1
        assert events[0]["type"] == "content_block_delta"
        assert events[0]["data"]["delta"]["text"] == "Hello"

    def test_tool_use_stream(self):
        """Test tool use content block events."""
        adapter = AnthropicStreamToCanonicalAdapter()

        # Tool use block start
        events = adapter.feed(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {}},
            },
        )
        assert len(events) == 1
        assert events[0]["type"] == "content_block_start"
        assert events[0]["data"]["type"] == "tool_use"
        assert events[0]["data"]["name"] == "get_weather"

        # Input JSON delta
        events = adapter.feed(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"city"'}},
        )
        assert len(events) == 1
        assert events[0]["data"]["delta"]["type"] == "tool_use_delta"
        assert events[0]["data"]["delta"]["partial_json"] == '{"city"'

    def test_message_stop(self):
        """Test message_stop event."""
        adapter = AnthropicStreamToCanonicalAdapter()

        events = adapter.feed("message_stop", {"type": "message_stop"})
        assert len(events) == 1
        assert events[0]["type"] == "message_stop"


class TestCanonicalStreamToOpenAIAdapter:
    """Tests for Canonical -> OpenAI streaming conversion."""

    def test_message_start_to_openai(self):
        """Test converting message_start to OpenAI format."""
        adapter = CanonicalStreamToOpenAIAdapter(model="gpt-4o")

        chunks = adapter.feed({"type": "message_start", "data": {"role": "assistant", "model": "gpt-4o"}})

        assert len(chunks) == 1
        assert chunks[0]["object"] == "chat.completion.chunk"
        assert chunks[0]["model"] == "gpt-4o"
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"

    def test_text_delta_to_openai(self):
        """Test converting text deltas to OpenAI format."""
        adapter = CanonicalStreamToOpenAIAdapter(model="gpt-4o")

        chunks = adapter.feed(
            {"type": "content_block_delta", "data": {"delta": {"type": "text_delta", "text": "Hello world"}}}
        )

        assert len(chunks) == 1
        assert chunks[0]["choices"][0]["delta"]["content"] == "Hello world"

    def test_tool_use_to_openai(self):
        """Test converting tool use to OpenAI format."""
        adapter = CanonicalStreamToOpenAIAdapter(model="gpt-4o")

        # Tool use block start
        chunks = adapter.feed(
            {"type": "content_block_start", "data": {"type": "tool_use", "id": "call_123", "name": "get_weather"}}
        )
        assert len(chunks) == 1
        assert "tool_calls" in chunks[0]["choices"][0]["delta"]
        assert chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_message_stop_to_openai(self):
        """Test converting message_stop to OpenAI format."""
        adapter = CanonicalStreamToOpenAIAdapter(model="gpt-4o")

        chunks = adapter.feed({"type": "message_stop", "data": {"stop_reason": "end_turn"}})

        assert len(chunks) == 1
        assert chunks[0]["choices"][0]["finish_reason"] == "stop"


class TestCanonicalStreamToAnthropicAdapter:
    """Tests for Canonical -> Anthropic streaming conversion."""

    def test_message_start_to_anthropic(self):
        """Test converting message_start to Anthropic format."""
        adapter = CanonicalStreamToAnthropicAdapter(model="claude-3-5-sonnet-20241022")

        events = adapter.feed(
            {"type": "message_start", "data": {"role": "assistant", "model": "claude-3-5-sonnet-20241022"}}
        )

        assert len(events) == 1
        assert events[0]["sse_type"] == "message_start"
        assert events[0]["data"]["message"]["role"] == "assistant"
        assert events[0]["data"]["message"]["model"] == "claude-3-5-sonnet-20241022"

    def test_text_delta_to_anthropic(self):
        """Test converting text deltas to Anthropic format."""
        adapter = CanonicalStreamToAnthropicAdapter(model="claude-3-5-sonnet-20241022")

        events = adapter.feed(
            {"type": "content_block_delta", "data": {"index": 0, "delta": {"type": "text_delta", "text": "Hello"}}}
        )

        assert len(events) == 1
        assert events[0]["sse_type"] == "content_block_delta"
        assert events[0]["data"]["delta"]["type"] == "text_delta"
        assert events[0]["data"]["delta"]["text"] == "Hello"

    def test_tool_use_to_anthropic(self):
        """Test converting tool use to Anthropic format."""
        adapter = CanonicalStreamToAnthropicAdapter(model="claude-3-5-sonnet-20241022")

        # Tool use block start
        events = adapter.feed(
            {"type": "content_block_start", "data": {"index": 0, "type": "tool_use", "id": "toolu_123", "name": "search"}}
        )
        assert len(events) == 1
        assert events[0]["sse_type"] == "content_block_start"
        assert events[0]["data"]["content_block"]["type"] == "tool_use"
        assert events[0]["data"]["content_block"]["name"] == "search"

    def test_message_stop_to_anthropic(self):
        """Test converting message_stop to Anthropic format."""
        adapter = CanonicalStreamToAnthropicAdapter(model="claude-3-5-sonnet-20241022")

        events = adapter.feed({"type": "message_stop", "data": {}})

        assert len(events) == 1
        assert events[0]["sse_type"] == "message_stop"


class TestCrossFormatStreaming:
    """Tests for cross-format streaming conversion."""

    def test_openai_to_anthropic_stream(self):
        """Test OpenAI -> Canonical -> Anthropic streaming."""
        openai_adapter = OpenAIStreamToCanonicalAdapter()
        anthropic_adapter = CanonicalStreamToAnthropicAdapter(model="claude-3-5-sonnet-20241022")

        # OpenAI chunk
        openai_chunk = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}],
        }

        # Convert through canonical
        canonical_events = openai_adapter.feed(openai_chunk)
        anthropic_events = []
        for event in canonical_events:
            anthropic_events.extend(anthropic_adapter.feed(event))

        # May include message_start + content_block_delta for empty content
        assert len(anthropic_events) >= 1
        assert anthropic_events[0]["sse_type"] == "message_start"

    def test_anthropic_to_openai_stream(self):
        """Test Anthropic -> Canonical -> OpenAI streaming."""
        anthropic_adapter = AnthropicStreamToCanonicalAdapter()
        openai_adapter = CanonicalStreamToOpenAIAdapter(model="gpt-4o")

        # Anthropic message start
        canonical_events = anthropic_adapter.feed(
            "message_start",
            {"type": "message_start", "message": {"id": "msg_123", "role": "assistant", "model": "claude-3-5-sonnet"}},
        )

        openai_chunks = []
        for event in canonical_events:
            openai_chunks.extend(openai_adapter.feed(event))

        assert len(openai_chunks) == 1
        assert openai_chunks[0]["object"] == "chat.completion.chunk"
        assert openai_chunks[0]["choices"][0]["delta"]["role"] == "assistant"
