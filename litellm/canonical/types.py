from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union


class CanonicalContentBlock(TypedDict, total=False):
    """
    Provider-agnostic content block used by canonical messages.

    type: "text", "tool_use", "tool_result", or provider-specific types.
    """

    type: str
    text: str
    name: str
    id: str
    input: Dict[str, Any]
    tool_use_id: str
    content: Union[str, List["CanonicalContentBlock"]]
    data: Any
    provider_fields: Dict[str, Any]


class CanonicalMessage(TypedDict, total=False):
    role: Literal["system", "user", "assistant", "tool"]
    content: List[CanonicalContentBlock]
    name: str
    tool_call_id: str
    provider_fields: Dict[str, Any]


class CanonicalTool(TypedDict, total=False):
    name: str
    description: str
    parameters: Dict[str, Any]
    provider_fields: Dict[str, Any]


class CanonicalRequest(TypedDict, total=False):
    model: str
    messages: List[CanonicalMessage]
    parameters: Dict[str, Any]
    stream: bool
    tools: List[CanonicalTool]
    tool_choice: Any
    metadata: Dict[str, Any]
    provider_passthrough: Dict[str, Any]


class CanonicalUsage(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CanonicalResponse(TypedDict, total=False):
    content: List[CanonicalContentBlock]
    usage: CanonicalUsage
    stop_reason: Optional[str]
    model: Optional[str]
    metadata: Dict[str, Any]
    provider_passthrough: Dict[str, Any]


class CanonicalStreamEvent(TypedDict, total=False):
    """
    Canonical streaming event shape. Intended to map to provider SSE formats.
    """

    type: str
    data: Dict[str, Any]
