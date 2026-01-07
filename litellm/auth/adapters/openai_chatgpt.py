from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from ..core import AuthRecord, AuthStatus, RequestContext
from .base import AdapterCapabilities
from .llms_oauth_loader import load_module
from .utils import token_dataclass_to_metadata

_logger = logging.getLogger(__name__)


class OpenAIChatGPTSubscriptionAdapter:
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

    def prepare(
        self, headers: Dict[str, str], ctx: RequestContext, auth: AuthRecord
    ) -> Dict[str, str]:
        token = auth.metadata.get("access_token")
        if not token:
            raise ValueError("missing access_token for openai subscription")
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
            raise ValueError("missing refresh_token for openai subscription")

        try:
            token = self._oauth().refresh_tokens(
                refresh_token=refresh_token,
                token_url=auth.attributes.get("token_url"),
                client_id=auth.attributes.get("client_id"),
                client_secret=auth.attributes.get("client_secret"),
                scopes=auth.attributes.get("scopes") or auth.attributes.get("scope"),
            )
        except Exception as e:
            # Extract retry-after if available for rate limit handling
            retry_after = None
            response = getattr(e, "response", None)
            if response is not None:
                retry_after = getattr(response, "headers", {}).get("Retry-After")
            if retry_after:
                _logger.debug("OpenAI refresh rate limited, retry-after: %s", retry_after)
            raise

        if not token:
            raise ValueError("refresh_tokens returned empty response for openai")

        updated = auth.clone()
        updated.metadata.update(token_dataclass_to_metadata(token))
        updated.last_refreshed_at = datetime.now(timezone.utc)
        updated.status = AuthStatus.ACTIVE
        updated.unavailable = False
        updated.status_message = ""
        updated.next_refresh_after = None  # Clear pending refresh
        return updated
