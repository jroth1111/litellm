"""
Unified /v1/messages endpoint - (Anthropic Spec)
"""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.integrations.custom_guardrail import ModifyResponseException
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    create_streaming_response,
)
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.types.utils import TokenCountResponse

router = APIRouter()

def _anthropic_error_type_for_status(status_code: int) -> str:
    if status_code == status.HTTP_400_BAD_REQUEST:
        return "invalid_request_error"
    if status_code == status.HTTP_401_UNAUTHORIZED:
        return "authentication_error"
    if status_code == status.HTTP_403_FORBIDDEN:
        return "permission_error"
    if status_code == status.HTTP_404_NOT_FOUND:
        return "not_found_error"
    if status_code == status.HTTP_429_TOO_MANY_REQUESTS:
        return "rate_limit_error"
    if status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        return "overloaded_error"
    if status_code == status.HTTP_504_GATEWAY_TIMEOUT:
        return "timeout_error"
    return "api_error"


def _anthropic_error_response(
    message: object,
    status_code: int,
    *,
    headers: Optional[dict] = None,
    error_type: Optional[str] = None,
) -> JSONResponse:
    payload = {
        "type": "error",
        "error": {
            "type": error_type or _anthropic_error_type_for_status(status_code),
            "message": str(message),
        },
    }
    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers=headers or {},
    )


@router.post(
    "/v1/messages",
    tags=["[beta] Anthropic `/v1/messages`"],
    dependencies=[Depends(user_api_key_auth)],
)
async def anthropic_response(  # noqa: PLR0915
    fastapi_response: Response,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Use `{PROXY_BASE_URL}/anthropic/v1/messages` instead - [Docs](https://docs.litellm.ai/docs/anthropic_completion).

    This was a BETA endpoint that calls 100+ LLMs in the anthropic format.
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data = await _read_request_body(request=request)
    base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        result = await base_llm_response_processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="anthropic_messages",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=None,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
        return result
    except ModifyResponseException as e:
        # Guardrail flagged content in passthrough mode - return 200 with violation message
        _data = e.request_data
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=_data,
        )

        # Create Anthropic-formatted response with violation message
        import uuid
        from litellm.types.utils import AnthropicMessagesResponse

        _anthropic_response = AnthropicMessagesResponse(
            id=f"msg_{str(uuid.uuid4())}",
            type="message",
            role="assistant",
            content=[{"type": "text", "text": e.message}],
            model=e.model,
            stop_reason="end_turn",
            usage={"input_tokens": 0, "output_tokens": 0},
        )

        if data.get("stream", None) is not None and data["stream"] is True:
            # For streaming, use the standard SSE data generator
            async def _passthrough_stream_generator():
                yield _anthropic_response

            selected_data_generator = (
                ProxyBaseLLMRequestProcessing.async_sse_data_generator(
                    response=_passthrough_stream_generator(),
                    user_api_key_dict=user_api_key_dict,
                    request_data=_data,
                    proxy_logging_obj=proxy_logging_obj,
                )
            )

            return await create_streaming_response(
                generator=selected_data_generator,
                media_type="text/event-stream",
                headers={},
            )

        return _anthropic_response
    except ProxyException as e:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        try:
            if e.code is not None:
                status_code = int(e.code)
        except Exception:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return _anthropic_error_response(
            getattr(e, "message", str(e)),
            status_code,
            headers=getattr(e, "headers", None),
            error_type=getattr(e, "type", None),
        )
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.anthropic_response(): Exception occured - {}".format(
                str(e)
            )
        )

        # Extract model_id from request metadata (same as success path)
        litellm_metadata = data.get("litellm_metadata", {}) or {}
        model_info = litellm_metadata.get("model_info", {}) or {}
        model_id = model_info.get("id", "") or ""

        # Get headers
        headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=user_api_key_dict,
            call_id=data.get("litellm_call_id", ""),
            model_id=model_id,
            version=version,
            response_cost=0,
            model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            request_data=data,
            timeout=getattr(e, "timeout", None),
            litellm_logging_obj=None,
        )

        error_msg = f"{str(e)}"
        status_code = getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
        return _anthropic_error_response(
            getattr(e, "message", error_msg),
            status_code,
            headers=headers,
        )


@router.post(
    "/v1/messages/count_tokens",
    tags=["[beta] Anthropic Messages Token Counting"],
    dependencies=[Depends(user_api_key_auth)],
)
async def count_tokens(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # Used for auth
):
    """
    Count tokens for Anthropic Messages API format.
    
    This endpoint follows the Anthropic Messages API token counting specification.
    It accepts the same parameters as the /v1/messages endpoint but returns
    token counts instead of generating a response.
    
    Example usage:
    ```
    curl -X POST "http://localhost:4000/v1/messages/count_tokens?beta=true" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer your-key" \
      -d '{
        "model": "claude-3-sonnet-20240229",
        "messages": [{"role": "user", "content": "Hello Claude!"}]
      }'
    ```
    
    Returns: {"input_tokens": <number>}
    """
    from litellm.proxy.proxy_server import token_counter as internal_token_counter

    try:
        request_data = await _read_request_body(request=request)
        data: dict = {**request_data}

        # Extract required fields
        model_name = data.get("model")
        messages = data.get("messages", [])

        if not model_name:
            raise HTTPException(
                status_code=400, detail={"error": "model parameter is required"}
            )

        if not messages:
            raise HTTPException(
                status_code=400, detail={"error": "messages parameter is required"}
            )

        # Create TokenCountRequest for the internal endpoint
        from litellm.proxy._types import TokenCountRequest

        token_request = TokenCountRequest(model=model_name, messages=messages)

        # Call the internal token counter function with direct request flag set to False
        token_response = await internal_token_counter(
            request=token_request,
            call_endpoint=True,
        )
        _token_response_dict: dict = {}
        if isinstance(token_response, TokenCountResponse):
            _token_response_dict = token_response.model_dump()
        elif isinstance(token_response, dict):
            _token_response_dict = token_response

        # Convert the internal response to Anthropic API format
        return {"input_tokens": _token_response_dict.get("total_tokens", 0)}
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.anthropic_endpoints.count_tokens(): Exception occurred - {}".format(
                str(e)
            )
        )
        raise HTTPException(
            status_code=500, detail={"error": f"Internal server error: {str(e)}"}
        )


@router.post(
    "/v1/complete",
    tags=["[beta] Anthropic `/v1/complete`"],
    dependencies=[Depends(user_api_key_auth)],
)
async def anthropic_complete(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Minimal Anthropic /v1/complete support (non-streaming).

    Maps Anthropic completion params to OpenAI /v1/completions format.
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    try:
        data = await _read_request_body(request=request)
        if data.get("stream"):
            raise HTTPException(
                status_code=400,
                detail={"error": "streaming not supported for /v1/complete"},
            )

        model = data.get("model")
        prompt = data.get("prompt")
        if not model or not prompt:
            raise HTTPException(
                status_code=400,
                detail={"error": "model and prompt are required"},
            )

        openai_data = {
            "model": model,
            "prompt": prompt,
            "max_tokens": data.get("max_tokens_to_sample") or data.get("max_tokens"),
            "temperature": data.get("temperature"),
            "top_p": data.get("top_p"),
            "stop": data.get("stop_sequences"),
        }
        openai_data = {k: v for k, v in openai_data.items() if v is not None}

        base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=openai_data)
        result = await base_llm_response_processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="atext_completion",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=None,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )

        if isinstance(result, Response):
            return result

        if hasattr(result, "model_dump"):
            payload = result.model_dump()
        elif hasattr(result, "dict"):
            payload = result.dict()  # type: ignore[call-arg]
        elif isinstance(result, dict):
            payload = result
        else:
            raise HTTPException(status_code=500, detail={"error": "unexpected response"})

        choices = payload.get("choices") or []
        completion_text = ""
        stop_reason = None
        if choices:
            choice0 = choices[0] or {}
            completion_text = (
                choice0.get("text")
                or (choice0.get("message") or {}).get("content")
                or ""
            )
            stop_reason = choice0.get("finish_reason")

        return {
            "id": payload.get("id") or f"compl_{uuid.uuid4().hex}",
            "model": payload.get("model") or model,
            "completion": completion_text,
            "stop_reason": stop_reason,
        }
    except ProxyException as e:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        try:
            if e.code is not None:
                status_code = int(e.code)
        except Exception:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return _anthropic_error_response(
            getattr(e, "message", str(e)),
            status_code,
            headers=getattr(e, "headers", None),
            error_type=getattr(e, "type", None),
        )
    except HTTPException as e:
        status_code = e.status_code or status.HTTP_400_BAD_REQUEST
        return _anthropic_error_response(
            getattr(e, "detail", str(e)),
            status_code,
        )
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.anthropic_endpoints.anthropic_complete(): Exception occurred - {}".format(
                str(e)
            )
        )
        return _anthropic_error_response(
            f"Internal server error: {str(e)}",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
