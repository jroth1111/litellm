from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from .types import (
    CanonicalContentBlock,
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalTool,
)

_OPENAI_PARAM_KEYS = {
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "n",
    "seed",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "response_format",
    "stream_options",
    "parallel_tool_calls",
    "tool_choice",
    "function_call",
    "functions",
    "reasoning_effort",
    "service_tier",
    "safety_identifier",
    "user",
}

_OPENAI_REQUEST_KEYS = {
    "model",
    "messages",
    "prompt",
    "tools",
    "tool_choice",
    "metadata",
    "stream",
} | _OPENAI_PARAM_KEYS

_ANTHROPIC_PARAM_KEYS = {
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "thinking",
    "container",
}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _safe_json_loads(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, (dict, list)):
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception:
            return {"raw": payload}
    return {"raw": payload}


def _safe_json_dumps(payload: Any) -> str:
    try:
        return json.dumps(payload)
    except Exception:
        return json.dumps({"raw": str(payload)})


def _openai_content_to_blocks(content: Any) -> List[CanonicalContentBlock]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        blocks: List[CanonicalContentBlock] = []
        for part in content:
            if isinstance(part, str):
                blocks.append({"type": "text", "text": part})
                continue
            if not isinstance(part, dict):
                blocks.append({"type": "text", "text": str(part)})
                continue
            part_type = str(part.get("type") or "text")
            if part_type == "text":
                blocks.append({"type": "text", "text": str(part.get("text") or "")})
            else:
                blocks.append({"type": part_type, "data": dict(part)})
        return blocks
    return [{"type": "text", "text": str(content)}]


def _blocks_to_openai_content(
    blocks: List[CanonicalContentBlock],
) -> Union[str, List[Dict[str, Any]]]:
    if not blocks:
        return ""
    text_blocks = [b for b in blocks if b.get("type") == "text"]
    non_text = [b for b in blocks if b.get("type") != "text"]
    if non_text:
        parts: List[Dict[str, Any]] = []
        for block in blocks:
            block_type = block.get("type") or "text"
            if block_type == "text":
                parts.append({"type": "text", "text": block.get("text") or ""})
            else:
                data = block.get("data")
                if isinstance(data, dict):
                    part = dict(data)
                    part.setdefault("type", block_type)
                    parts.append(part)
                else:
                    parts.append({"type": block_type, "data": data})
        return parts
    if len(text_blocks) == 1:
        return text_blocks[0].get("text") or ""
    return [{"type": "text", "text": b.get("text") or ""} for b in text_blocks]


def _openai_tool_calls_to_blocks(tool_calls: Any) -> List[CanonicalContentBlock]:
    blocks: List[CanonicalContentBlock] = []
    for tool_call in _as_list(tool_calls):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        name = function.get("name")
        args = function.get("arguments")
        blocks.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or str(uuid.uuid4()),
                "name": name or "",
                "input": _safe_json_loads(args),
            }
        )
    return blocks


def _blocks_to_openai_tool_calls(
    blocks: List[CanonicalContentBlock],
) -> List[Dict[str, Any]]:
    tool_calls: List[Dict[str, Any]] = []
    for block in blocks:
        if block.get("type") != "tool_use":
            continue
        name = block.get("name") or ""
        args = block.get("input")
        tool_calls.append(
            {
                "id": block.get("id") or str(uuid.uuid4()),
                "type": "function",
                "function": {"name": name, "arguments": _safe_json_dumps(args)},
            }
        )
    return tool_calls


def _split_blocks(
    blocks: List[CanonicalContentBlock],
) -> Tuple[List[CanonicalContentBlock], List[CanonicalContentBlock], List[CanonicalContentBlock]]:
    text_blocks: List[CanonicalContentBlock] = []
    tool_use_blocks: List[CanonicalContentBlock] = []
    tool_result_blocks: List[CanonicalContentBlock] = []
    for block in blocks:
        block_type = block.get("type") or ""
        if block_type == "tool_use":
            tool_use_blocks.append(block)
        elif block_type == "tool_result":
            tool_result_blocks.append(block)
        else:
            text_blocks.append(block)
    return text_blocks, tool_use_blocks, tool_result_blocks


def openai_request_to_canonical(data: Dict[str, Any]) -> CanonicalRequest:
    messages = data.get("messages")
    if not messages and "prompt" in data:
        messages = [{"role": "user", "content": data.get("prompt")}]
    canonical_messages: List[CanonicalMessage] = []
    for msg in _as_list(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content_blocks = _openai_content_to_blocks(msg.get("content"))
        tool_blocks = _openai_tool_calls_to_blocks(msg.get("tool_calls"))
        function_call = msg.get("function_call")
        if function_call:
            tool_blocks.extend(
                _openai_tool_calls_to_blocks(
                    [{"id": str(uuid.uuid4()), "function": function_call}]
                )
            )
        canonical_messages.append(
            {
                "role": role,
                "content": content_blocks + tool_blocks,
            }
        )
    tools: List[CanonicalTool] = []
    for tool in _as_list(data.get("tools")):
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            tools.append({"provider_fields": dict(tool)})
            continue
        fn = tool.get("function") or {}
        tools.append(
            {
                "name": fn.get("name") or "",
                "description": fn.get("description") or "",
                "parameters": fn.get("parameters") or {},
            }
        )

    parameters: Dict[str, Any] = {}
    provider_passthrough: Dict[str, Any] = {}
    for key, value in data.items():
        if key in _OPENAI_PARAM_KEYS:
            parameters[key] = value
        elif key not in _OPENAI_REQUEST_KEYS:
            provider_passthrough[key] = value

    return CanonicalRequest(
        model=str(data.get("model") or ""),
        messages=canonical_messages,
        parameters=parameters,
        stream=bool(data.get("stream") or False),
        tools=tools,
        tool_choice=data.get("tool_choice"),
        metadata=data.get("metadata") or {},
        provider_passthrough=provider_passthrough,
    )


def canonical_to_openai_request(req: CanonicalRequest) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []
    for msg in req.get("messages", []):
        blocks = msg.get("content") or []
        text_blocks, tool_use_blocks, tool_result_blocks = _split_blocks(blocks)
        role = msg.get("role") or "user"

        if role == "tool":
            for block in tool_result_blocks or text_blocks:
                messages.append(
                    {
                        "role": "tool",
                        "content": block.get("content") or block.get("text") or "",
                        "tool_call_id": block.get("tool_use_id") or msg.get("tool_call_id"),
                    }
                )
            continue

        message: Dict[str, Any] = {"role": role}
        if text_blocks:
            message["content"] = _blocks_to_openai_content(text_blocks)
        else:
            message["content"] = ""
        tool_calls = _blocks_to_openai_tool_calls(tool_use_blocks)
        if tool_calls:
            message["tool_calls"] = tool_calls
        messages.append(message)

        for block in tool_result_blocks:
            messages.append(
                {
                    "role": "tool",
                    "content": block.get("content") or block.get("text") or "",
                    "tool_call_id": block.get("tool_use_id"),
                }
            )

    tools: List[Dict[str, Any]] = []
    for tool in req.get("tools", []) or []:
        name = tool.get("name")
        if name:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description") or "",
                        "parameters": tool.get("parameters") or {},
                    },
                }
            )
        else:
            provider_fields = tool.get("provider_fields")
            if isinstance(provider_fields, dict):
                tools.append(dict(provider_fields))

    data: Dict[str, Any] = {
        "model": req.get("model") or "",
        "messages": messages,
    }
    if req.get("stream"):
        data["stream"] = True
    if tools:
        data["tools"] = tools
    if req.get("tool_choice") is not None:
        data["tool_choice"] = req.get("tool_choice")
    if req.get("metadata"):
        data["metadata"] = req.get("metadata")

    params = req.get("parameters") or {}
    for key, value in params.items():
        if key not in data:
            data[key] = value

    passthrough = req.get("provider_passthrough") or {}
    for key, value in passthrough.items():
        if key not in data:
            data[key] = value

    return data


def anthropic_request_to_canonical(data: Dict[str, Any]) -> CanonicalRequest:
    messages = data.get("messages") or []
    canonical_messages: List[CanonicalMessage] = []
    system = data.get("system")
    if isinstance(system, str) and system:
        canonical_messages.append(
            {"role": "system", "content": [{"type": "text", "text": system}]}
        )
    for msg in _as_list(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        blocks: List[CanonicalContentBlock] = []
        if isinstance(content, str):
            blocks = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    blocks.append({"type": "text", "text": str(part)})
                    continue
                part_type = str(part.get("type") or "text")
                if part_type == "text":
                    blocks.append({"type": "text", "text": str(part.get("text") or "")})
                elif part_type == "tool_use":
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": part.get("id") or str(uuid.uuid4()),
                            "name": part.get("name") or "",
                            "input": part.get("input") or {},
                        }
                    )
                elif part_type == "tool_result":
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": part.get("tool_use_id") or "",
                            "content": part.get("content") or "",
                        }
                    )
                else:
                    blocks.append({"type": part_type, "data": dict(part)})
        canonical_messages.append({"role": role, "content": blocks})

    tools: List[CanonicalTool] = []
    for tool in _as_list(data.get("tools")):
        if not isinstance(tool, dict):
            continue
        tools.append(
            {
                "name": tool.get("name") or "",
                "description": tool.get("description") or "",
                "parameters": tool.get("input_schema") or tool.get("parameters") or {},
            }
        )

    parameters: Dict[str, Any] = {}
    provider_passthrough: Dict[str, Any] = {}
    for key, value in data.items():
        if key in _ANTHROPIC_PARAM_KEYS:
            parameters[key] = value
        elif key not in {"model", "messages", "system", "tools", "tool_choice", "metadata", "stream"}:
            provider_passthrough[key] = value

    return CanonicalRequest(
        model=str(data.get("model") or ""),
        messages=canonical_messages,
        parameters=parameters,
        stream=bool(data.get("stream") or False),
        tools=tools,
        tool_choice=data.get("tool_choice"),
        metadata=data.get("metadata") or {},
        provider_passthrough=provider_passthrough,
    )


def canonical_to_anthropic_request(req: CanonicalRequest) -> Dict[str, Any]:
    system_parts: List[str] = []
    messages: List[Dict[str, Any]] = []
    for msg in req.get("messages", []):
        role = msg.get("role") or "user"
        blocks = msg.get("content") or []
        text_blocks, tool_use_blocks, tool_result_blocks = _split_blocks(blocks)
        if role == "system":
            for block in text_blocks:
                system_parts.append(block.get("text") or "")
            continue

        content_blocks: List[Dict[str, Any]] = []
        for block in text_blocks:
            content_blocks.append({"type": "text", "text": block.get("text") or ""})
        for block in tool_use_blocks:
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": block.get("id") or str(uuid.uuid4()),
                    "name": block.get("name") or "",
                    "input": block.get("input") or {},
                }
            )
        if content_blocks:
            messages.append({"role": role, "content": content_blocks})

        if tool_result_blocks:
            tool_result_content: List[Dict[str, Any]] = []
            for block in tool_result_blocks:
                tool_result_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.get("tool_use_id") or "",
                        "content": block.get("content") or block.get("text") or "",
                    }
                )
            messages.append({"role": "user", "content": tool_result_content})

    tools: List[Dict[str, Any]] = []
    for tool in req.get("tools", []) or []:
        name = tool.get("name")
        if not name:
            provider_fields = tool.get("provider_fields")
            if isinstance(provider_fields, dict):
                tools.append(dict(provider_fields))
            continue
        tools.append(
            {
                "name": name,
                "description": tool.get("description") or "",
                "input_schema": tool.get("parameters") or {},
            }
        )

    data: Dict[str, Any] = {
        "model": req.get("model") or "",
        "messages": messages,
    }
    if system_parts:
        data["system"] = "\n".join([part for part in system_parts if part])
    if req.get("stream"):
        data["stream"] = True
    if tools:
        data["tools"] = tools
    if req.get("tool_choice") is not None:
        data["tool_choice"] = req.get("tool_choice")
    if req.get("metadata"):
        data["metadata"] = req.get("metadata")

    params = req.get("parameters") or {}
    for key, value in params.items():
        if key not in data:
            data[key] = value

    passthrough = req.get("provider_passthrough") or {}
    for key, value in passthrough.items():
        if key not in data:
            data[key] = value

    return data


def openai_response_to_canonical(response: Any) -> CanonicalResponse:
    if response is None:
        return CanonicalResponse(content=[])
    if isinstance(response, dict):
        payload = response
    elif hasattr(response, "model_dump"):
        payload = response.model_dump()
    elif hasattr(response, "dict"):
        payload = response.dict()  # type: ignore[call-arg]
    else:
        return CanonicalResponse(content=[], provider_passthrough={"raw": str(response)})

    choices = payload.get("choices") or []
    message = {}
    finish_reason = None
    if choices:
        choice0 = choices[0] or {}
        message = choice0.get("message") or {}
        finish_reason = choice0.get("finish_reason")

    content_blocks = _openai_content_to_blocks(message.get("content"))
    content_blocks.extend(_openai_tool_calls_to_blocks(message.get("tool_calls")))
    function_call = message.get("function_call")
    if function_call:
        content_blocks.extend(
            _openai_tool_calls_to_blocks(
                [{"id": str(uuid.uuid4()), "function": function_call}]
            )
        )

    return CanonicalResponse(
        content=content_blocks,
        usage=payload.get("usage") or {},
        stop_reason=finish_reason,
        model=payload.get("model"),
        metadata={},
        provider_passthrough={},
    )


def anthropic_response_to_canonical(response: Any) -> CanonicalResponse:
    """Convert an Anthropic API response to canonical format."""
    if response is None:
        return CanonicalResponse(content=[])
    if isinstance(response, dict):
        payload = response
    elif hasattr(response, "model_dump"):
        payload = response.model_dump()
    elif hasattr(response, "dict"):
        payload = response.dict()  # type: ignore[call-arg]
    else:
        return CanonicalResponse(content=[], provider_passthrough={"raw": str(response)})

    content_blocks: List[CanonicalContentBlock] = []
    for block in _as_list(payload.get("content")):
        if not isinstance(block, dict):
            content_blocks.append({"type": "text", "text": str(block)})
            continue
        block_type = str(block.get("type") or "text")
        if block_type == "text":
            content_blocks.append({"type": "text", "text": str(block.get("text") or "")})
        elif block_type == "tool_use":
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": block.get("id") or str(uuid.uuid4()),
                    "name": block.get("name") or "",
                    "input": block.get("input") or {},
                }
            )
        elif block_type == "tool_result":
            content_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.get("tool_use_id") or "",
                    "content": block.get("content") or "",
                }
            )
        elif block_type == "thinking":
            content_blocks.append(
                {
                    "type": "thinking",
                    "text": block.get("thinking") or "",
                    "data": dict(block),
                }
            )
        else:
            content_blocks.append({"type": block_type, "data": dict(block)})

    # Map Anthropic usage to canonical usage
    usage = payload.get("usage") or {}
    canonical_usage: Dict[str, Any] = {}
    if "input_tokens" in usage:
        canonical_usage["prompt_tokens"] = usage["input_tokens"]
    if "output_tokens" in usage:
        canonical_usage["completion_tokens"] = usage["output_tokens"]
    if canonical_usage.get("prompt_tokens") and canonical_usage.get("completion_tokens"):
        canonical_usage["total_tokens"] = (
            canonical_usage["prompt_tokens"] + canonical_usage["completion_tokens"]
        )
    # Preserve any additional usage fields
    for key, value in usage.items():
        if key not in ("input_tokens", "output_tokens"):
            canonical_usage[key] = value

    return CanonicalResponse(
        content=content_blocks,
        usage=canonical_usage,
        stop_reason=payload.get("stop_reason"),
        model=payload.get("model"),
        metadata={},
        provider_passthrough={
            "id": payload.get("id"),
            "type": payload.get("type"),
        },
    )


def canonical_to_openai_response(response: CanonicalResponse) -> Dict[str, Any]:
    """Convert a canonical response to OpenAI API format."""
    content_blocks = response.get("content", []) or []
    text_blocks, tool_use_blocks, _ = _split_blocks(content_blocks)

    # Build message content
    message: Dict[str, Any] = {"role": "assistant"}
    if text_blocks:
        message["content"] = _blocks_to_openai_content(text_blocks)
    else:
        message["content"] = None

    # Add tool calls if present
    tool_calls = _blocks_to_openai_tool_calls(tool_use_blocks)
    if tool_calls:
        message["tool_calls"] = tool_calls

    # Map stop_reason to finish_reason
    stop_reason = response.get("stop_reason")
    finish_reason = _anthropic_stop_reason_to_openai(stop_reason) if stop_reason else None

    # Map canonical usage to OpenAI usage
    usage = response.get("usage") or {}
    openai_usage: Dict[str, Any] = {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
    if not openai_usage["total_tokens"]:
        openai_usage["total_tokens"] = (
            openai_usage["prompt_tokens"] + openai_usage["completion_tokens"]
        )

    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(__import__("time").time()),
        "model": response.get("model") or "",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": openai_usage,
    }


def _anthropic_stop_reason_to_openai(stop_reason: Optional[str]) -> Optional[str]:
    """Map Anthropic stop_reason to OpenAI finish_reason."""
    if not stop_reason:
        return None
    mapping = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }
    return mapping.get(stop_reason, stop_reason)


def canonical_to_anthropic_response(response: CanonicalResponse) -> Dict[str, Any]:
    content_blocks: List[Dict[str, Any]] = []
    for block in response.get("content", []) or []:
        block_type = block.get("type") or "text"
        if block_type == "text":
            content_blocks.append({"type": "text", "text": block.get("text") or ""})
        elif block_type == "tool_use":
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": block.get("id") or str(uuid.uuid4()),
                    "name": block.get("name") or "",
                    "input": block.get("input") or {},
                }
            )
        elif block_type == "tool_result":
            content_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.get("tool_use_id") or "",
                    "content": block.get("content") or block.get("text") or "",
                }
            )
        else:
            data = block.get("data")
            if isinstance(data, dict):
                payload = dict(data)
                payload.setdefault("type", block_type)
                content_blocks.append(payload)
            else:
                content_blocks.append({"type": block_type, "data": data})

    usage = response.get("usage") or {}
    usage_payload: Dict[str, Any] = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }
    for key, value in usage.items():
        if key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            continue
        usage_payload[key] = value

    return {
        "id": "msg_" + uuid.uuid4().hex,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": response.get("model") or "",
        "stop_reason": response.get("stop_reason"),
        "usage": usage_payload,
    }
