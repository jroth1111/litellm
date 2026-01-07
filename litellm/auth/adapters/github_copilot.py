from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict


from ..core import AuthRecord, RequestContext
from .base import AdapterCapabilities, BaseSubscriptionAdapter
from .llms_oauth_loader import load_module
from .utils import token_dataclass_to_metadata

_logger = logging.getLogger(__name__)


class GitHubCopilotSubscriptionAdapter(BaseSubscriptionAdapter):
    """GitHub Copilot subscription OAuth adapter with device code flow."""
    
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
        except Exception as e:
            _logger.debug("Failed to fetch user info for github_copilot: %s", e)
        return meta

    def supports(self, model: str) -> bool:
        """Check if this adapter supports the given model.
        
        Matches models starting with 'github_copilot/' or 'copilot'.
        """
        lowered = (model or "").lower()
        if lowered.startswith("github_copilot/"):
            return True
        # Direct copilot model names without other provider prefix
        if "/" not in lowered and lowered.startswith("copilot"):
            return True
        return False

    def refresh(self, auth: AuthRecord, ctx: RequestContext) -> AuthRecord:
        raise ValueError("github_copilot subscription tokens are not refreshable; re-login required")
