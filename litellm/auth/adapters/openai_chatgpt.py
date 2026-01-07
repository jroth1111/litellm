from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Optional

from ..core import AuthRecord, RequestContext
from .base import AdapterCapabilities, BaseSubscriptionAdapter
from .llms_oauth_loader import load_module
from .utils import token_dataclass_to_metadata


class OpenAIChatGPTSubscriptionAdapter(BaseSubscriptionAdapter):
    """
    OpenAI ChatGPT Plus / Codex consumer OAuth adapter.

    - CLI: browser-based OAuth2 with PKCE
    - Router: Bearer auth + refresh token rotation
    """

    provider = "openai"
    supports_refresh = True
    refresh_lead_default = timedelta(minutes=5)
    capabilities = AdapterCapabilities(
        login_flow="browser_pkce",
        supports_refresh=True,
        supports_models_list=True,
    )

    @staticmethod
    def _oauth():
        return load_module(
            "llms/openai/codex_oauth.py", "litellm.auth.adapters._openai_codex_oauth"
        )

    @property
    def default_redirect_uri(self) -> str:
        return str(self._oauth().OPENAI_REDIRECT_URI)

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
        
        Matches:
        - Models explicitly prefixed with 'openai/'
        - Direct OpenAI model names (gpt-4*, gpt-5*, chatgpt*, o1*, o3*)
        
        Does NOT match:
        - Azure models (azure/gpt-4)
        - Other provider-prefixed models
        """
        lowered = (model or "").lower()
        # Explicit openai/ prefix
        if lowered.startswith("openai/"):
            return True
        # Direct model names without other provider prefix
        if "/" not in lowered:
            openai_patterns = ("gpt-4", "gpt-5", "chatgpt", "o1-", "o1_", "o3-", "o3_")
            if any(lowered.startswith(p) or p in lowered for p in openai_patterns):
                return True
        return False

    def refresh(self, auth: AuthRecord, ctx: RequestContext) -> AuthRecord:
        refresh_token = auth.metadata.get("refresh_token")
        if not refresh_token:
            raise ValueError("missing refresh_token for openai subscription")

        try:
            # OpenAI-specific: passes scopes param
            token = self._oauth().refresh_tokens(
                refresh_token=refresh_token,
                token_url=auth.attributes.get("token_url"),
                client_id=auth.attributes.get("client_id"),
                client_secret=auth.attributes.get("client_secret"),
                scopes=auth.attributes.get("scopes") or auth.attributes.get("scope"),
            )
        except Exception as e:
            self._extract_retry_after(e)
            raise

        self._validate_token(token)
        return self._finalize_refresh(auth, token)
