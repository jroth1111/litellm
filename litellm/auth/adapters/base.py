from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LoginFlow = Literal["browser_pkce", "device_code", "cursor_poll"]


@dataclass(frozen=True)
class AdapterCapabilities:
    """
    Describes what a subscription adapter can do.

    `login_flow` guides the CLI on how to perform `litellm auth login <provider>`.
    """

    login_flow: LoginFlow
    supports_refresh: bool = True
    supports_models_list: bool = False

