from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from ..core import AuthKind, AuthRecord, AuthStatus, RequestContext
from ..selector import interpret_status_code, retry_after_from_exception

__all__ = [
    "AdapterCapabilities",
    "BaseSubscriptionAdapter",
    "FailureClassification",
    "LoginFlow",
    "LoginStart",
]

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


@dataclass(frozen=True)
class LoginStart:
    method: LoginFlow
    url: Optional[str] = None
    instructions: str = ""
    state: Optional[str] = None
    redirect_uri: Optional[str] = None
    device_code: Optional[Any] = None
    user_code: Optional[str] = None
    session: Optional[Any] = None


@dataclass(frozen=True)
class FailureClassification:
    rotatable: bool
    cooldown_ms: Optional[int] = None
    is_auth_expired: bool = False
    reason: str = ""


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
    supports_api_key: bool = True
    supports_wellknown: bool = True
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

    def methods(self) -> List[Dict[str, Any]]:
        """
        List supported auth methods for this provider.
        """
        methods: List[Dict[str, Any]] = [
            {
                "type": "oauth",
                "label": "OAuth",
                "flow": getattr(self.capabilities, "login_flow", None),
            }
        ]
        if self.supports_api_key:
            methods.append({"type": "api", "label": "API key"})
        if self.supports_wellknown:
            methods.append({"type": "wellknown", "label": "Well-known env"})
        return methods

    def start_login(
        self,
        *,
        state: Optional[str] = None,
        code_challenge: Optional[str] = None,
        redirect_uri: Optional[str] = None,
    ) -> LoginStart:
        """
        Begin an OAuth login flow and return a structured login session.
        """
        flow = getattr(self.capabilities, "login_flow", None)
        if flow == "device_code":
            device_authorize = getattr(self, "device_authorize", None)
            if device_authorize is None:
                raise ValueError(f"{self.provider} does not support device_code login")
            device = device_authorize()
            url = getattr(device, "verification_uri_complete", None) or getattr(
                device, "verification_uri", None
            )
            instructions = "Open the URL and enter the code to continue."
            return LoginStart(
                method="device_code",
                url=str(url) if url else None,
                instructions=instructions,
                device_code=device,
                user_code=getattr(device, "user_code", None),
            )

        if flow == "browser_pkce":
            if not state or not code_challenge:
                raise ValueError("state and code_challenge are required for browser_pkce")
            final_redirect = redirect_uri or getattr(self, "default_redirect_uri", None)
            if not final_redirect:
                raise ValueError("redirect_uri is required for browser_pkce")
            authorize_url = getattr(self, "authorize_url", None)
            if authorize_url is None:
                raise ValueError(f"{self.provider} does not support authorize_url")
            url = authorize_url(
                state=state,
                code_challenge=code_challenge,
                redirect_uri=final_redirect,
            )
            return LoginStart(
                method="browser_pkce",
                url=str(url),
                instructions="Open the URL to continue the login flow.",
                state=state,
                redirect_uri=final_redirect,
            )

        raise NotImplementedError(f"{self.provider} login flow not implemented: {flow}")
    
    def prepare(
        self, headers: Dict[str, str], ctx: RequestContext, auth: AuthRecord
    ) -> Dict[str, str]:
        """
        Add Bearer authorization header.
        
        Identical across all adapters - extracts access_token from auth.metadata.
        """
        token = auth.resolve_secret()
        if not token:
            if auth.kind == AuthKind.WELLKNOWN:
                env_key = auth.metadata.get("env_key") or auth.attributes.get("env_key")
                raise ValueError(
                    f"missing well-known token for {self.provider} (env_key={env_key})"
                )
            raise ValueError(f"missing access_token for {self.provider}")
        new_headers = dict(headers)
        new_headers["Authorization"] = f"Bearer {token}"
        return new_headers

    def classify_failure(
        self,
        error: BaseException,
        *,
        status_code: Optional[int] = None,
    ) -> FailureClassification:
        """
        Classify an error for rotation/cooldown decisions.
        """
        code = status_code if status_code is not None else interpret_status_code(error)
        retry_after = retry_after_from_exception(error)
        cooldown_ms = (
            int(retry_after.total_seconds() * 1000) if retry_after is not None else None
        )

        if code in (401, 403):
            return FailureClassification(
                rotatable=True,
                cooldown_ms=cooldown_ms,
                is_auth_expired=True,
                reason="auth_error",
            )
        if code == 429:
            return FailureClassification(
                rotatable=True,
                cooldown_ms=cooldown_ms,
                is_auth_expired=False,
                reason="rate_limit",
            )
        if code is not None and code >= 500:
            return FailureClassification(
                rotatable=True,
                cooldown_ms=cooldown_ms,
                is_auth_expired=False,
                reason="server_error",
            )

        error_name = type(error).__name__.lower()
        if "timeout" in error_name:
            return FailureClassification(
                rotatable=True,
                cooldown_ms=cooldown_ms,
                is_auth_expired=False,
                reason="timeout",
            )

        return FailureClassification(
            rotatable=False,
            cooldown_ms=None,
            is_auth_expired=False,
            reason="non_retryable",
        )
    
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

    def _parse_retry_after(self, exception: Exception) -> Optional[timedelta]:
        """
        Extract and parse Retry-After header into a timedelta.

        Handles both:
        - Seconds format: "120" -> timedelta(seconds=120)
        - HTTP-date format: "Wed, 21 Oct 2015 07:28:00 GMT" -> timedelta from now

        Returns None if header is missing or unparseable.
        """
        retry_after = self._extract_retry_after(exception)
        if not retry_after:
            return None

        # Try parsing as integer seconds
        try:
            seconds = int(retry_after)
            return timedelta(seconds=max(1, seconds))
        except ValueError:
            pass

        # Try parsing as HTTP-date
        try:
            from email.utils import parsedate_to_datetime

            retry_dt = parsedate_to_datetime(retry_after)
            now = datetime.now(timezone.utc)
            delta = retry_dt - now
            if delta.total_seconds() > 0:
                return delta
            return timedelta(seconds=1)  # Already passed, minimal backoff
        except (ValueError, TypeError):
            pass

        _logger.debug(
            "%s: could not parse Retry-After value: %s", self.provider, retry_after
        )
        return None

    def _handle_refresh_error(
        self, exception: Exception, auth: AuthRecord, default_backoff_seconds: int = 300
    ) -> AuthRecord:
        """
        Handle a refresh error by extracting Retry-After and setting backoff.

        If a Retry-After header is present, use it to set next_refresh_after.
        Otherwise, use the default backoff.

        Args:
            exception: The exception from the refresh attempt
            auth: The AuthRecord to update
            default_backoff_seconds: Default backoff if no Retry-After header

        Returns:
            Updated AuthRecord with appropriate backoff set
        """
        retry_delta = self._parse_retry_after(exception)
        if retry_delta is None:
            retry_delta = timedelta(seconds=default_backoff_seconds)

        # Cap at reasonable maximum (1 hour)
        max_backoff = timedelta(hours=1)
        if retry_delta > max_backoff:
            retry_delta = max_backoff

        updated = auth.clone()
        updated.next_refresh_after = datetime.now(timezone.utc) + retry_delta
        updated.status_message = f"Refresh failed: {str(exception)[:200]}"
        _logger.debug(
            "%s: refresh failed, backoff for %s", self.provider, retry_delta
        )
        return updated
    
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
