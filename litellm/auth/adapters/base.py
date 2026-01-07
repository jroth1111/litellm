from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Literal, Optional

from ..core import AuthRecord, AuthStatus, RequestContext

__all__ = ["AdapterCapabilities", "BaseSubscriptionAdapter", "LoginFlow"]

_logger = logging.getLogger(__name__)

LoginFlow = Literal["browser_pkce", "device_code", "cursor_poll"]


@dataclass(frozen=True)
class AdapterCapabilities:
    """
    Describes what a subscription adapter can do.

    `login_flow` guides the CLI on how to perform `litellm auth login <provider>`.
    """

    login_flow: LoginFlow
    supports_refresh: bool = True
    supports_models_list: bool = False


class BaseSubscriptionAdapter(ABC):
    """
    Abstract base class for subscription OAuth adapters.
    
    Provides shared implementations for common patterns across all adapters:
    - prepare(): Bearer token injection
    - expiration(): Token expiration lookup
    - refresh_lead(): Refresh lead time
    - _finalize_refresh(): Post-refresh AuthRecord update
    - _extract_retry_after(): Retry-After header extraction from exceptions
    
    Subclasses must implement:
    - provider: str class attribute
    - supports(): Model matching logic (idiosyncratic per provider)
    - _oauth(): Module loader for provider-specific OAuth helpers
    """
    
    provider: str  # Must be defined by subclass
    supports_refresh: bool = True
    refresh_lead_default: timedelta = timedelta(minutes=5)
    capabilities: AdapterCapabilities  # Must be defined by subclass
    
    @staticmethod
    @abstractmethod
    def _oauth():
        """Load the provider-specific OAuth helper module."""
        raise NotImplementedError
    
    @abstractmethod
    def supports(self, model: str) -> bool:
        """Check if this adapter supports the given model."""
        raise NotImplementedError
    
    @abstractmethod
    def refresh(self, auth: AuthRecord, ctx: RequestContext) -> AuthRecord:
        """Refresh expired tokens. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def prepare(
        self, headers: Dict[str, str], ctx: RequestContext, auth: AuthRecord
    ) -> Dict[str, str]:
        """
        Add Bearer authorization header.
        
        Identical across all adapters - extracts access_token from auth.metadata.
        """
        token = auth.metadata.get("access_token")
        if not token:
            raise ValueError(f"missing access_token for {self.provider} subscription")
        new_headers = dict(headers)
        new_headers["Authorization"] = f"Bearer {token}"
        return new_headers
    
    def expiration(self, auth: AuthRecord) -> Optional[datetime]:
        """Get token expiration time. Delegates to AuthRecord.expiration_time()."""
        return auth.expiration_time()
    
    def refresh_lead(self, auth: AuthRecord) -> Optional[timedelta]:
        """Get refresh lead time. Returns the class default."""
        return self.refresh_lead_default
    
    def _extract_retry_after(self, exception: Exception) -> Optional[str]:
        """
        Extract Retry-After header from an HTTP exception response.
        
        Returns the header value if present, None otherwise.
        Logs debug message if retry-after is found.
        """
        retry_after = None
        response = getattr(exception, "response", None)
        if response is not None:
            retry_after = getattr(response, "headers", {}).get("Retry-After")
        if retry_after:
            _logger.debug("%s refresh rate limited, retry-after: %s", self.provider, retry_after)
        return retry_after
    
    def _validate_token(self, token: Any, operation: str = "refresh") -> None:
        """Validate that token response is not empty."""
        if not token:
            raise ValueError(f"{operation}_tokens returned empty response for {self.provider}")
    
    def _finalize_refresh(self, auth: AuthRecord, token: Any) -> AuthRecord:
        """
        Finalize a refresh operation by updating the AuthRecord.
        
        This is the shared post-refresh logic:
        - Clone the auth record
        - Update metadata with new token
        - Update timestamps and status
        - Clear pending refresh flag
        
        Args:
            auth: The original AuthRecord
            token: The new token dataclass from refresh_tokens()
            
        Returns:
            Updated AuthRecord clone
        """
        # Import here to avoid circular dependency
        from .utils import token_dataclass_to_metadata
        
        updated = auth.clone()
        updated.metadata.update(token_dataclass_to_metadata(token))
        updated.last_refreshed_at = datetime.now(timezone.utc)
        updated.status = AuthStatus.ACTIVE
        updated.unavailable = False
        updated.status_message = ""
        updated.next_refresh_after = None  # Clear pending refresh
        return updated
