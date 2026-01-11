from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from .types import CanonicalStreamEvent

__all__ = [
    "OpenAIStreamToCanonicalAdapter",
    "AnthropicStreamToCanonicalAdapter",
    "CanonicalStreamToOpenAIAdapter",
    "CanonicalStreamToAnthropicAdapter",
]


class OpenAIStreamToCanonicalAdapter:
    """
    Best-effort adapter that converts OpenAI streaming chunks to canonical events.
    """

    def __init__(self) -> None:
        self._started = False
        self._model: Optional[str] = None

    def feed(self, chunk: Any) -> List[CanonicalStreamEvent]:
        payload: Dict[str, Any]
        if isinstance(chunk, dict):
            payload = chunk
        elif hasattr(chunk, "model_dump"):
            payload = chunk.model_dump()
        elif hasattr(chunk, "dict"):
            payload = chunk.dict()  # type: ignore[call-arg]
        else:
            return []

        events: List[CanonicalStreamEvent] = []

        # Track model for later use
        if payload.get("model"):
            self._model = payload.get("model")

        choices = payload.get("choices") or []
        if not choices:
            return events
        delta = (choices[0] or {}).get("delta") or {}
        finish_reason = (choices[0] or {}).get("finish_reason")

        if not self._started and delta.get("role"):
            self._started = True
            events.append({
                "type": "message_start",
                "data": {
                    "role": delta.get("role"),
                    "model": self._model,
                }
            })

        if "content" in delta and delta.get("content") is not None:
            events.append(
                {
                    "type": "content_block_delta",
                    "data": {"delta": {"type": "text_delta", "text": delta.get("content")}},
                }
            )

        if delta.get("tool_calls"):
            for tool_call in delta.get("tool_calls", []):
                events.append(
                    {
                        "type": "content_block_delta",
                        "data": {
                            "delta": {
                                "type": "tool_use_delta",
                                "index": tool_call.get("index", 0),
                                "id": tool_call.get("id"),
                                "name": (tool_call.get("function") or {}).get("name"),
                                "arguments": (tool_call.get("function") or {}).get("arguments"),
                            }
                        },
                    }
                )

        if finish_reason:
            events.append({"type": "message_stop", "data": {"stop_reason": finish_reason}})

        return events


class AnthropicStreamToCanonicalAdapter:
    """
    Adapter that converts Anthropic streaming events to canonical events.

    Anthropic SSE format:
    - message_start: Contains message metadata
    - content_block_start: New content block begins
    - content_block_delta: Content updates
    - content_block_stop: Content block ends
    - message_delta: Final message metadata updates
    - message_stop: Stream complete
    """

    def __init__(self) -> None:
        self._model: Optional[str] = None
        self._current_block_index: int = 0
        self._current_block_type: Optional[str] = None
        self._tool_use_id: Optional[str] = None
        self._tool_name: Optional[str] = None

    def feed(self, event_type: str, data: Any) -> List[CanonicalStreamEvent]:
        """
        Feed an Anthropic SSE event and return canonical events.

        Args:
            event_type: The SSE event type (e.g., "message_start", "content_block_delta")
            data: The parsed JSON data from the event
        """
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return []

        if not isinstance(data, dict):
            return []

        events: List[CanonicalStreamEvent] = []

        if event_type == "message_start":
            message = data.get("message", {})
            self._model = message.get("model")
            events.append({
                "type": "message_start",
                "data": {
                    "role": message.get("role", "assistant"),
                    "model": self._model,
                    "id": message.get("id"),
                    "usage": message.get("usage"),
                }
            })

        elif event_type == "content_block_start":
            self._current_block_index = data.get("index", 0)
            content_block = data.get("content_block", {})
            self._current_block_type = content_block.get("type")

            if self._current_block_type == "tool_use":
                self._tool_use_id = content_block.get("id")
                self._tool_name = content_block.get("name")
                events.append({
                    "type": "content_block_start",
                    "data": {
                        "index": self._current_block_index,
                        "type": "tool_use",
                        "id": self._tool_use_id,
                        "name": self._tool_name,
                    }
                })
            elif self._current_block_type == "text":
                events.append({
                    "type": "content_block_start",
                    "data": {
                        "index": self._current_block_index,
                        "type": "text",
                    }
                })
            elif self._current_block_type == "thinking":
                events.append({
                    "type": "content_block_start",
                    "data": {
                        "index": self._current_block_index,
                        "type": "thinking",
                    }
                })

        elif event_type == "content_block_delta":
            delta = data.get("delta", {})
            delta_type = delta.get("type")

            if delta_type == "text_delta":
                events.append({
                    "type": "content_block_delta",
                    "data": {
                        "index": data.get("index", self._current_block_index),
                        "delta": {
                            "type": "text_delta",
                            "text": delta.get("text", ""),
                        }
                    }
                })
            elif delta_type == "input_json_delta":
                events.append({
                    "type": "content_block_delta",
                    "data": {
                        "index": data.get("index", self._current_block_index),
                        "delta": {
                            "type": "tool_use_delta",
                            "partial_json": delta.get("partial_json", ""),
                        }
                    }
                })
            elif delta_type == "thinking_delta":
                events.append({
                    "type": "content_block_delta",
                    "data": {
                        "index": data.get("index", self._current_block_index),
                        "delta": {
                            "type": "thinking_delta",
                            "thinking": delta.get("thinking", ""),
                        }
                    }
                })

        elif event_type == "content_block_stop":
            events.append({
                "type": "content_block_stop",
                "data": {"index": data.get("index", self._current_block_index)}
            })

        elif event_type == "message_delta":
            delta = data.get("delta", {})
            events.append({
                "type": "message_delta",
                "data": {
                    "stop_reason": delta.get("stop_reason"),
                    "usage": data.get("usage"),
                }
            })

        elif event_type == "message_stop":
            events.append({
                "type": "message_stop",
                "data": {}
            })

        elif event_type == "ping":
            # Ping events can be ignored or passed through
            pass

        elif event_type == "error":
            events.append({
                "type": "error",
                "data": data
            })

        return events


class CanonicalStreamToOpenAIAdapter:
    """
    Adapter that converts canonical streaming events to OpenAI SSE format.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self._model = model or ""
        self._id = "chatcmpl-" + uuid.uuid4().hex[:24]
        self._created = int(__import__("time").time())
        self._tool_call_indices: Dict[str, int] = {}
        self._next_tool_index = 0

    def feed(self, event: CanonicalStreamEvent) -> List[Dict[str, Any]]:
        """
        Convert a canonical stream event to OpenAI format chunks.
        """
        chunks: List[Dict[str, Any]] = []
        event_type = event.get("type", "")
        data = event.get("data", {})

        if event_type == "message_start":
            # OpenAI sends role in the first chunk
            self._model = data.get("model") or self._model
            chunks.append(self._make_chunk(
                delta={"role": data.get("role", "assistant"), "content": ""},
            ))

        elif event_type == "content_block_start":
            block_type = data.get("type")
            if block_type == "tool_use":
                tool_id = data.get("id") or str(uuid.uuid4())
                tool_name = data.get("name", "")
                index = self._next_tool_index
                self._tool_call_indices[tool_id] = index
                self._next_tool_index += 1
                chunks.append(self._make_chunk(
                    delta={
                        "tool_calls": [{
                            "index": index,
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": ""},
                        }]
                    }
                ))

        elif event_type == "content_block_delta":
            delta_data = data.get("delta", {})
            delta_type = delta_data.get("type")

            if delta_type == "text_delta":
                text = delta_data.get("text", "")
                if text:
                    chunks.append(self._make_chunk(delta={"content": text}))

            elif delta_type == "tool_use_delta":
                partial_json = delta_data.get("partial_json") or delta_data.get("arguments", "")
                tool_id = delta_data.get("id")
                index = delta_data.get("index", 0)

                if tool_id and tool_id in self._tool_call_indices:
                    index = self._tool_call_indices[tool_id]

                if partial_json:
                    chunks.append(self._make_chunk(
                        delta={
                            "tool_calls": [{
                                "index": index,
                                "function": {"arguments": partial_json},
                            }]
                        }
                    ))

        elif event_type == "message_delta":
            stop_reason = data.get("stop_reason")
            if stop_reason:
                finish_reason = self._map_stop_reason(stop_reason)
                chunks.append(self._make_chunk(delta={}, finish_reason=finish_reason))

        elif event_type == "message_stop":
            stop_reason = data.get("stop_reason")
            finish_reason = self._map_stop_reason(stop_reason) if stop_reason else "stop"
            chunks.append(self._make_chunk(delta={}, finish_reason=finish_reason))

        return chunks

    def _make_chunk(
        self,
        delta: Dict[str, Any],
        finish_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "id": self._id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self._model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }

    def _map_stop_reason(self, stop_reason: str) -> str:
        mapping = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
        }
        return mapping.get(stop_reason, stop_reason)


class CanonicalStreamToAnthropicAdapter:
    """
    Adapter that converts canonical streaming events to Anthropic SSE format.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self._model = model or ""
        self._id = "msg_" + uuid.uuid4().hex

    def feed(self, event: CanonicalStreamEvent) -> List[Dict[str, Any]]:
        """
        Convert a canonical stream event to Anthropic format events.

        Returns a list of (event_type, data) tuples for SSE emission.
        """
        events: List[Dict[str, Any]] = []
        event_type = event.get("type", "")
        data = event.get("data", {})

        if event_type == "message_start":
            self._model = data.get("model") or self._model
            events.append({
                "sse_type": "message_start",
                "data": {
                    "type": "message_start",
                    "message": {
                        "id": data.get("id") or self._id,
                        "type": "message",
                        "role": data.get("role", "assistant"),
                        "content": [],
                        "model": self._model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": data.get("usage") or {"input_tokens": 0, "output_tokens": 0},
                    }
                }
            })

        elif event_type == "content_block_start":
            block_type = data.get("type", "text")
            index = data.get("index", 0)

            if block_type == "tool_use":
                events.append({
                    "sse_type": "content_block_start",
                    "data": {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": data.get("id") or str(uuid.uuid4()),
                            "name": data.get("name", ""),
                            "input": {},
                        }
                    }
                })
            else:
                events.append({
                    "sse_type": "content_block_start",
                    "data": {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": block_type, "text": ""}
                    }
                })

        elif event_type == "content_block_delta":
            delta_data = data.get("delta", {})
            delta_type = delta_data.get("type")
            index = data.get("index", 0)

            if delta_type == "text_delta":
                events.append({
                    "sse_type": "content_block_delta",
                    "data": {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "text_delta",
                            "text": delta_data.get("text", ""),
                        }
                    }
                })
            elif delta_type == "tool_use_delta":
                partial_json = delta_data.get("partial_json") or delta_data.get("arguments", "")
                events.append({
                    "sse_type": "content_block_delta",
                    "data": {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": partial_json,
                        }
                    }
                })

        elif event_type == "content_block_stop":
            events.append({
                "sse_type": "content_block_stop",
                "data": {
                    "type": "content_block_stop",
                    "index": data.get("index", 0),
                }
            })

        elif event_type == "message_delta":
            events.append({
                "sse_type": "message_delta",
                "data": {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": data.get("stop_reason"),
                        "stop_sequence": None,
                    },
                    "usage": data.get("usage") or {"output_tokens": 0},
                }
            })

        elif event_type == "message_stop":
            events.append({
                "sse_type": "message_stop",
                "data": {"type": "message_stop"}
            })

        return events
