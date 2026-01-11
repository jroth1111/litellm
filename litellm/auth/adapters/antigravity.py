from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, Optional

from ..core import AuthRecord, RequestContext
from .base import AdapterCapabilities, BaseSubscriptionAdapter
from .llms_oauth_loader import load_module
from .utils import token_dataclass_to_metadata

_logger = logging.getLogger(__name__)


class AntigravitySubscriptionAdapter(BaseSubscriptionAdapter):
    """Antigravity (Google AI) subscription OAuth adapter."""
    
    provider = "antigravity"
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
            "llms/antigravity/antigravity_oauth.py",
            "litellm.auth.adapters._antigravity_oauth",
        )

    @property
    def default_redirect_uri(self) -> str:
        return str(self._oauth().ANTIGRAVITY_REDIRECT_URI)

    def authorize_url(
        self, *, state: str, code_challenge: str, redirect_uri: str
    ) -> str:
        return str(
            self._oauth().generate_auth_url(
                state=state, redirect_uri=redirect_uri, code_challenge=code_challenge
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
            state=state,
            expected_state=expected_state,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier,
        )
        meta: Dict[str, Any] = token_dataclass_to_metadata(tokens)
        
        # Antigravity-specific: enrich with user info and project ID
        try:
            info = self._oauth().fetch_user_info(meta.get("access_token") or "")
            if isinstance(info, dict) and info.get("email"):
                meta["email"] = info.get("email")
        except Exception as e:
            _logger.debug("Failed to fetch user info for antigravity: %s", e)
        try:
            proj = self._oauth().fetch_project_id(meta.get("access_token") or "")
            if proj:
                meta["project_id"] = proj
        except Exception as e:
            _logger.debug("Failed to fetch project_id for antigravity: %s", e)
        return meta

    def supports(self, model: str) -> bool:
        """Check if this adapter supports the given model.
        
        Matches models starting with 'antigravity' (case-insensitive).
        Examples: 'antigravity', 'antigravity/model', 'Antigravity-Pro'
        """
        lowered = (model or "").lower()
        return lowered.startswith("antigravity")

    def refresh(self, auth: AuthRecord, ctx: RequestContext) -> AuthRecord:
        refresh_token = auth.metadata.get("refresh_token")
        if not refresh_token:
            raise ValueError("missing refresh_token for antigravity subscription")

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
