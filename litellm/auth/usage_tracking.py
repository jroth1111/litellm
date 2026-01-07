"""
Token usage metrics tracking for OAuth AuthRecords.

This module provides utilities to update usage counters on AuthRecords
after API calls, enabling observability of token utilization.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .core import AuthRecord


def track_auth_usage(
    auth: AuthRecord,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    success: bool = True,
) -> AuthRecord:
    """
    Update usage counters on an AuthRecord after an API call.

    This creates a clone with updated usage metrics. The caller is
    responsible for persisting the updated record to the store.

    Args:
        auth: The AuthRecord to update.
        prompt_tokens: Input tokens consumed.
        completion_tokens: Output tokens generated.
        success: Whether the API call succeeded (False increments error_count).

    Returns:
        Updated AuthRecord clone with incremented usage metrics.
    """
    updated = auth.clone()
    updated.request_count += 1
    updated.prompt_tokens += prompt_tokens
    updated.completion_tokens += completion_tokens
    updated.last_request_at = datetime.now(timezone.utc)

    if not success:
        updated.error_count += 1

    return updated


def record_api_call(
    auth: AuthRecord,
    *,
    model: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    success: bool = True,
    error_message: Optional[str] = None,
) -> AuthRecord:
    """
    Record an API call with full context (model, error details).

    Extended version of track_auth_usage with additional metadata.

    Args:
        auth: The AuthRecord to update.
        model: The model used for the API call.
        prompt_tokens: Input tokens consumed.
        completion_tokens: Output tokens generated.
        success: Whether the API call succeeded.
        error_message: Error message if call failed.

    Returns:
        Updated AuthRecord clone.
    """
    updated = track_auth_usage(
        auth,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        success=success,
    )

    # Optionally store last error for debugging
    if not success and error_message:
        updated.status_message = f"Last error: {error_message[:200]}"

    return updated


def reset_usage_counters(auth: AuthRecord) -> AuthRecord:
    """
    Reset all usage counters on an AuthRecord.

    Useful for starting a new billing period or after manual rotation.

    Args:
        auth: The AuthRecord to reset.

    Returns:
        Updated AuthRecord clone with zeroed counters.
    """
    updated = auth.clone()
    updated.request_count = 0
    updated.error_count = 0
    updated.prompt_tokens = 0
    updated.completion_tokens = 0
    updated.last_request_at = None
    return updated
