from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from ..core import AuthRecord, AuthStatus, RequestContext
from .base import AdapterCapabilities
from .llms_oauth_loader import load_module
from .utils import token_dataclass_to_metadata


class CursorSubscriptionAdapter:
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

    def start_login(self):
        return self._oauth().start_cursor_login()

    def poll_login(self, session, *, timeout_seconds: int) -> Dict[str, Any]:
        token = self._oauth().poll_for_token(session, max_wait_seconds=timeout_seconds)
        return token_dataclass_to_metadata(token)

    def supports(self, model: str) -> bool:
        lowered = (model or "").lower()
        return lowered.startswith("cursor/") or lowered.startswith("cursor-") or lowered == "cursor"

    def prepare(
        self, headers: Dict[str, str], ctx: RequestContext, auth: AuthRecord
    ) -> Dict[str, str]:
        token = auth.metadata.get("access_token")
        if not token:
            raise ValueError("missing access_token for cursor subscription")
        new_headers = dict(headers)
        new_headers["Authorization"] = f"Bearer {token}"
        return new_headers

    def expiration(self, auth: AuthRecord) -> Optional[datetime]:
        return auth.expiration_time()

    def refresh_lead(self, auth: AuthRecord) -> Optional[timedelta]:
        return self.refresh_lead_default

    def refresh(self, auth: AuthRecord, ctx: RequestContext) -> AuthRecord:
        refresh_token = auth.metadata.get("refresh_token")
        if not refresh_token:
            raise ValueError("missing refresh_token for cursor subscription")

        token = self._oauth().refresh_tokens(
            refresh_token=refresh_token,
            token_url=auth.attributes.get("token_url"),
        )
        updated = auth.clone()
        updated.metadata.update(token_dataclass_to_metadata(token))
        updated.last_refreshed_at = datetime.now(timezone.utc)
        updated.status = AuthStatus.ACTIVE
        updated.unavailable = False
        updated.status_message = ""
        return updated

