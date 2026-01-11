from __future__ import annotations

import logging
import uuid
from datetime import timedelta
from typing import Any, Dict


from ..core import AuthRecord, RequestContext
from .base import AdapterCapabilities, BaseSubscriptionAdapter
from .llms_oauth_loader import load_module
from .utils import token_dataclass_to_metadata

_logger = logging.getLogger(__name__)

# GitHub Copilot requires specific headers for proper request handling
COPILOT_EDITOR_VERSION = "vscode/1.85.0"
COPILOT_EDITOR_PLUGIN_VERSION = "copilot/1.0.0"


class GitHubCopilotSubscriptionAdapter(BaseSubscriptionAdapter):
    """GitHub Copilot subscription OAuth adapter with device code flow."""

    provider = "github_copilot"
    supports_refresh = False
    supports_api_key = False
    supports_wellknown = False
    refresh_lead_default = timedelta(minutes=5)
    capabilities = AdapterCapabilities(
        login_flow="device_code",
        supports_refresh=False,
        supports_models_list=True,
    )

    @staticmethod
    def _oauth():
        return load_module(
            "llms/github_copilot/oauth_subscription.py",
            "litellm.auth.adapters._github_copilot_oauth_subscription",
        )

    def prepare(
        self, headers: Dict[str, str], ctx: RequestContext, auth: AuthRecord
    ) -> Dict[str, str]:
        """
        Add GitHub Copilot-specific headers.

        GitHub Copilot requires:
        - Authorization: Bearer <token>
        - X-GitHub-Api-Version: API version header
        - Editor-Version: Editor identifier
        - Editor-Plugin-Version: Plugin version
        - X-Request-Id: Request tracking ID
        - Copilot-Integration-Id: Integration identifier
        """
        token = auth.resolve_secret()
        if not token:
            raise ValueError(f"missing auth token for {self.provider}")

        new_headers = dict(headers)
        new_headers["Authorization"] = f"Bearer {token}"

        # Add GitHub Copilot specific headers
        if "X-GitHub-Api-Version" not in new_headers:
            new_headers["X-GitHub-Api-Version"] = "2023-11-28"

        if "Editor-Version" not in new_headers:
            editor_version = auth.attributes.get("editor_version", COPILOT_EDITOR_VERSION)
            new_headers["Editor-Version"] = editor_version

        if "Editor-Plugin-Version" not in new_headers:
            plugin_version = auth.attributes.get(
                "editor_plugin_version", COPILOT_EDITOR_PLUGIN_VERSION
            )
            new_headers["Editor-Plugin-Version"] = plugin_version

        # Add request tracking ID
        if "X-Request-Id" not in new_headers:
            new_headers["X-Request-Id"] = str(uuid.uuid4())

        # Add integration ID if specified
        integration_id = auth.attributes.get("copilot_integration_id")
        if integration_id and "Copilot-Integration-Id" not in new_headers:
            new_headers["Copilot-Integration-Id"] = integration_id

        # Add OpenAI-specific headers for compatibility
        if "OpenAI-Intent" not in new_headers:
            new_headers["OpenAI-Intent"] = "conversation-panel"

        return new_headers

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
        normalized = lowered.replace("-", "_")
        if normalized.startswith("github_copilot/"):
            return True
        if normalized.startswith("copilot/"):
            return True
        # Direct copilot model names without other provider prefix
        if "/" not in normalized and normalized.startswith("github_copilot"):
            return True
        if "/" not in normalized and normalized.startswith("copilot"):
            return True
        return False

    def refresh(self, auth: AuthRecord, ctx: RequestContext) -> AuthRecord:
        raise ValueError("github_copilot subscription tokens are not refreshable; re-login required")
