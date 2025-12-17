from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from ..core import AuthRecord, AuthStatus, RequestContext
from .base import AdapterCapabilities
from .llms_oauth_loader import load_module
from .utils import token_dataclass_to_metadata


class GitHubCopilotSubscriptionAdapter:
    provider = "github_copilot"
    supports_refresh = False
    refresh_lead_default = timedelta(minutes=5)
    capabilities = AdapterCapabilities(
        login_flow="device_code",
        supports_refresh=False,
        supports_models_list=False,
    )

    @staticmethod
    def _oauth():
        return load_module(
            "llms/github_copilot/oauth_subscription.py",
            "litellm.auth.adapters._github_copilot_oauth_subscription",
        )

    def device_authorize(self):
        return self._oauth().request_device_code()

    def device_poll(self, device_code, *, timeout_seconds: int) -> Dict[str, Any]:
        token = self._oauth().poll_for_token(device_code, max_wait_seconds=timeout_seconds)
        meta: Dict[str, Any] = token_dataclass_to_metadata(token)
        try:
            meta["account"] = self._oauth().fetch_user_info(meta.get("access_token") or "")
        except Exception:
            pass
        return meta

    def supports(self, model: str) -> bool:
        lowered = (model or "").lower()
        return lowered.startswith("github_copilot/") or "copilot" in lowered

    def prepare(
        self, headers: Dict[str, str], ctx: RequestContext, auth: AuthRecord
    ) -> Dict[str, str]:
        token = auth.metadata.get("access_token")
        if not token:
            raise ValueError("missing access_token for github_copilot subscription")
        new_headers = dict(headers)
        new_headers["Authorization"] = f"Bearer {token}"
        return new_headers

    def expiration(self, auth: AuthRecord) -> Optional[datetime]:
        return auth.expiration_time()

    def refresh_lead(self, auth: AuthRecord) -> Optional[timedelta]:
        return self.refresh_lead_default

    def refresh(self, auth: AuthRecord, ctx: RequestContext) -> AuthRecord:
        raise ValueError("github_copilot subscription tokens are not refreshable; re-login required")

