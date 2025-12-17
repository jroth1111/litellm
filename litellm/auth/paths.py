"""
Filesystem path helpers for auth/token storage.

Security posture:
- Default to user-scoped config directories (avoid writing secrets into repos).
- Keep the logic dependency-free (no platformdirs).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _xdg_config_home() -> Path:
    env = os.getenv("XDG_CONFIG_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".config"


def default_auth_store_dir() -> str:
    """
    Default directory for persisted AuthRecords.
    """
    return str(_xdg_config_home() / "litellm" / "auth")


def default_token_store_dir() -> str:
    """
    Default directory for ephemeral OAuth/device-flow tokens.
    """
    return str(_xdg_config_home() / "litellm" / "auth-tokens")


def find_git_root(path: str) -> Optional[str]:
    """
    Returns the nearest parent containing a `.git` directory/file, else None.
    """
    p = Path(path).expanduser()
    try:
        p = p.resolve()
    except Exception:
        p = p.absolute()
    for parent in (p, *p.parents):
        git = parent / ".git"
        if git.exists():
            return str(parent)
    return None

