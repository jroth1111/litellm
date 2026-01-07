from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Optional

from ..core import AuthRecord, RequestContext
from .base import AdapterCapabilities, BaseSubscriptionAdapter
from .llms_oauth_loader import load_module
from .utils import token_dataclass_to_metadata


class AnthropicSubscriptionAdapter(BaseSubscriptionAdapter):
    """Anthropic Claude subscription OAuth adapter."""
    
    provider = "anthropic"
    supports_refresh = True
    refresh_lead_default = timedelta(minutes=5)
    capabilities = AdapterCapabilities(
        login_flow="browser_pkce",
        supports_refresh=True,
        supports_models_list=False,
    )

    @staticmethod
    def _oauth():
        return load_module(
            "llms/anthropic/claude_oauth.py",
            "litellm.auth.adapters._anthropic_claude_oauth",
        )

    @property
    def default_redirect_uri(self) -> str:
        return str(self._oauth().ANTHROPIC_REDIRECT_URI)

    def authorize_url(
        self, *, state: str, code_challenge: str, redirect_uri: str
    ) -> str:
        return str(
            self._oauth().generate_auth_url(
                state=state, code_challenge=code_challenge, redirect_uri=redirect_uri
            )
        )

    def exchange_code(
        self,
        *,
        code: str,
        code_verifier: str,
        state: str,
        expected_state: Optional[str],
        redirect_uri: str,
    ) -> Dict[str, Any]:
        tokens = self._oauth().exchange_code_for_tokens(
            code=code,
            code_verifier=code_verifier,
            state=state,
            expected_state=expected_state,
            redirect_uri=redirect_uri,
        )
        return token_dataclass_to_metadata(tokens)

    def supports(self, model: str) -> bool:
        """Check if this adapter supports the given model.
        
        Matches models starting with 'anthropic/' or 'claude'.
        """
        lowered = (model or "").lower()
        if lowered.startswith("anthropic/"):
            return True
        # Direct claude model names without other provider prefix
        if "/" not in lowered and lowered.startswith("claude"):
            return True
        return False

    def refresh(self, auth: AuthRecord, ctx: RequestContext) -> AuthRecord:
        refresh_token = auth.metadata.get("refresh_token")
        if not refresh_token:
            raise ValueError("missing refresh_token for anthropic subscription")

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
