from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Optional

from ..core import AuthRecord, RequestContext
from .base import AdapterCapabilities, BaseSubscriptionAdapter, LoginStart
from .llms_oauth_loader import load_module
from .utils import token_dataclass_to_metadata


class CursorSubscriptionAdapter(BaseSubscriptionAdapter):
    """Cursor subscription OAuth adapter with polling-based login."""
    
    provider = "cursor"
    supports_refresh = True
    refresh_lead_default = timedelta(minutes=5)
    capabilities = AdapterCapabilities(
        login_flow="cursor_poll",
        supports_refresh=True,
        supports_models_list=True,
    )

    @staticmethod
    def _oauth():
        return load_module(
            "llms/cursor/oauth_subscription.py",
            "litellm.auth.adapters._cursor_oauth_subscription",
        )

    def start_login(
        self,
        *,
        state: Optional[str] = None,
        code_challenge: Optional[str] = None,
        redirect_uri: Optional[str] = None,
    ) -> LoginStart:
        session = self._oauth().start_cursor_login()
        login_url = getattr(session, "login_url", None) or getattr(session, "login", None)
        return LoginStart(
            method="cursor_poll",
            url=str(login_url) if login_url else None,
            instructions="Open the URL to continue the login flow.",
            session=session,
        )

    def poll_login(self, session, *, timeout_seconds: int) -> Dict[str, Any]:
        token = self._oauth().poll_for_token(session, max_wait_seconds=timeout_seconds)
        return token_dataclass_to_metadata(token)

    def supports(self, model: str) -> bool:
        """Check if this adapter supports the given model.
        
        Matches models starting with 'cursor/' or 'cursor-' or bare 'cursor'.
        """
        lowered = (model or "").lower()
        return lowered.startswith("cursor/") or lowered.startswith("cursor-") or lowered == "cursor"

    def refresh(self, auth: AuthRecord, ctx: RequestContext) -> AuthRecord:
        refresh_token = auth.metadata.get("refresh_token")
        if not refresh_token:
            raise ValueError("missing refresh_token for cursor subscription")

        try:
            token = self._oauth().refresh_tokens(
                refresh_token=refresh_token,
                token_url=auth.attributes.get("token_url"),
                client_id=auth.attributes.get("client_id"),
                client_secret=auth.attributes.get("client_secret"),
            )
        except Exception as e:
            self._extract_retry_after(e)
            raise

        self._validate_token(token)
        return self._finalize_refresh(auth, token)
