from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict

from ..core import AuthRecord, RequestContext
from .base import AdapterCapabilities, BaseSubscriptionAdapter
from .llms_oauth_loader import load_module
from .utils import token_dataclass_to_metadata


class QwenSubscriptionAdapter(BaseSubscriptionAdapter):
    """Qwen subscription OAuth adapter with device code flow."""
    
    provider = "qwen"
    supports_refresh = True
    refresh_lead_default = timedelta(minutes=5)
    capabilities = AdapterCapabilities(
        login_flow="device_code",
        supports_refresh=True,
        supports_models_list=True,
    )

    @staticmethod
    def _oauth():
        return load_module(
            "llms/qwen/oauth_subscription.py",
            "litellm.auth.adapters._qwen_oauth_subscription",
        )

    def device_authorize(self):
        return self._oauth().initiate_device_flow()

    def device_poll(self, flow, *, timeout_seconds: int) -> Dict[str, Any]:
        token = self._oauth().poll_for_token(
            flow.device_code,
            flow.code_verifier,
            initial_interval=float(getattr(flow, "interval", 5) or 5.0),
            max_wait_seconds=timeout_seconds,
        )
        return token_dataclass_to_metadata(token)

    def supports(self, model: str) -> bool:
        """Check if this adapter supports the given model.
        
        Matches models starting with 'qwen/' or 'qwen'.
        """
        lowered = (model or "").lower()
        if lowered.startswith("qwen/"):
            return True
        if lowered.startswith("dashscope/qwen"):
            return True
        # Direct qwen model names without other provider prefix
        if "/" not in lowered and lowered.startswith("qwen"):
            return True
        if "/" not in lowered and "qwen" in lowered:
            return True
        return False

    def refresh(self, auth: AuthRecord, ctx: RequestContext) -> AuthRecord:
        refresh_token = auth.metadata.get("refresh_token")
        if not refresh_token:
            raise ValueError("missing refresh_token for qwen subscription")

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
